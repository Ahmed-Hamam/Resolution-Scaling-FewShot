# miniImageNet — Plain ProtoNet (MobileNetV3-Small backbone) — LOCAL/VS CODE VERSION
# يشمل: حجمي الصورة 84x84 و 224x224، بدون LearnableNorm وبدون gamma
# (ProtoNet قياسي: النموذج الأولي = متوسط سمات الدعم، والمسافة = Euclidean
# التربيعية القياسية بدون أي معاملات تعلم إضافية)، مع تكرار التجربة على 3 seeds.
# miniImageNet: 100 صنف (64 train / 16 val / 20 test) — تقسيم Ravi & Larochelle القياسي.
#
# ⚠️ الفرق الوحيد عن نسخة Kaggle: تحميل الداتاست. بدل قراءة /kaggle/input
#    (غير موجود على الجهاز المحلي)، الكود بيحمّل الداتاست تلقائيًا بـ
#    kagglehub، ويحفظ كل حاجة في مجلد محلي بدل /kaggle/working. باقي الكود
#    **زي ما هو 100% بدون أي تغيير** — نفس الـ transforms الموحّدة، نفس
#    batch sizes حسب الدقة، نفس seed التقسيم=42.
#
# MATCHED-EPISODE EVALUATION:
# بالإضافة للتقييم العشوائي المعتاد، نولّد مجموعة ثابتة من تعريفات الـ episodes
# (نفس الأصناف + نفس صور الدعم/الاستعلام) مرة واحدة لكل k-shot، ونقيّم عليها
# نموذجي 84x84 و224x224 (لكل seed). بما إن الـ episodes مطابقة تمامًا، الفرق
# Δ_i = Acc224_i - Acc84_i لكل episode يصبح مقارنة "paired"، ونطبّق عليه
# paired t-test و Wilcoxon signed-rank test لاختبار H0: E[Δ] = 0.

import os
import json
try:
    import kagglehub
except ImportError:
    print("📦 Installing kagglehub...")
    os.system('pip install -q kagglehub')
    import kagglehub

try:
    from thop import profile as thop_profile
except ImportError:
    print("📦 Installing thop...")
    os.system('pip install -q thop')
    from thop import profile as thop_profile

try:
    from scipy import stats as scipy_stats
except ImportError:
    print("📦 Installing scipy...")
    os.system('pip install -q scipy')
    from scipy import stats as scipy_stats

import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# ============================================
# 0. LOCAL PATHS
# ============================================
# غيّر ده لو اسم الداتاست بتاعك على Kaggle مختلف.
MINI_KAGGLEHUB_ID = "arjunashok33/miniimagenet"  # عدّله لو مختلف عندك

WORK_ROOT       = os.path.join(os.getcwd(), "miniimagenet_workdir")
MINI_SPLIT_ROOT = os.path.join(WORK_ROOT, "miniimagenet_split")
os.makedirs(WORK_ROOT, exist_ok=True)

# ============================================
# 1. SETUP
# ============================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Processing on: {device}")

# Simple stage-timing registry
STAGE_TIMES = {}

class StageTimer:
    """Context manager that records wall-clock time for a named stage."""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"\n⏱️  [{self.name}] started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.t0
        STAGE_TIMES[self.name] = elapsed
        mins = elapsed / 60.0
        print(f"⏱️  [{self.name}] finished in {elapsed:.1f}s ({mins:.2f} min)")


def print_stage_time_summary():
    if not STAGE_TIMES:
        print("No stage timings recorded yet.")
        return
    print("\n" + "="*40)
    print("⏱️  STAGE TIME SUMMARY")
    print("="*40)
    total = 0.0
    for name, secs in STAGE_TIMES.items():
        print(f"  {name:45s} {secs:8.1f}s  ({secs/60:.2f} min)")
        total += secs
    print(f"  {'TOTAL':45s} {total:8.1f}s  ({total/60:.2f} min)")

# ============================================
# 2. MODEL PROFILING: PARAMETERS & FLOPs
# ============================================

def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_flops(backbone_module, input_size, device=device):
    """Backbone-only FLOPs/params via thop (feature extraction pass only)."""
    backbone_module = backbone_module.to(device).eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)

    class _FeatureOnly(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)

    wrapper = _FeatureOnly(backbone_module).to(device).eval()
    with torch.no_grad():
        macs, params = thop_profile(wrapper, inputs=(dummy,), verbose=False)

    gflops = macs / 1e9
    params_m = params / 1e6
    print(f"📐 Backbone profile @ {input_size}x{input_size}: "
          f"{params_m:.2f}M params | {gflops:.4f} GFLOPs (MACs)")
    return {"params": params, "params_M": params_m, "macs": macs, "gflops": gflops}


def report_full_model_stats(model, input_size):
    """Total params of ProtoNet wrapper (backbone only, no extra params) + FLOPs."""
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"\n📊 ProtoNet full model: "
          f"{total_params/1e6:.3f}M total params | "
          f"{trainable_params/1e6:.3f}M trainable params")
    flops_info = compute_flops(model.backbone, input_size=input_size)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        **flops_info
    }

# ============================================
# 3. DATA PREPARATION — miniImageNet (auto-download via kagglehub)
# ============================================
# miniImageNet: 100 classes sampled from ImageNet, 600 images/class.
# Standard split (Ravi & Larochelle 2017): 64 train / 16 val / 20 test classes.

def _find_class_dirs(root):
    """Return a directory whose immediate children are all class-folders
    (heuristic: at least 50 subdirs, each containing images)."""
    for r, dirs, files in os.walk(root):
        if not dirs:
            continue
        sub_dirs = [d for d in dirs if os.path.isdir(os.path.join(r, d))]
        if len(sub_dirs) < 50:  # miniImageNet has 100 classes total / per split >=16
            continue
        sample = os.path.join(r, sub_dirs[0])
        has_imgs = any(f.lower().endswith(('jpg', 'jpeg', 'png'))
                       for f in os.listdir(sample))
        if has_imgs:
            return r
    return None


def prepare_miniimagenet():
    if os.path.exists(MINI_SPLIT_ROOT):
        print("✅ miniImageNet split already present.")
        return

    print("\n📥 Downloading miniImageNet...")
    try:
        path = kagglehub.dataset_download(MINI_KAGGLEHUB_ID)
        dataset_path = path
        print(f"📁 Dataset downloaded to: {dataset_path}")

        os.makedirs(MINI_SPLIT_ROOT, exist_ok=True)

        # الحالة 1: الداتاست موزّع بالفعل بمجلدات train/val/test رسمية
        existing_splits = {}
        for split_name in ["train", "val", "test"]:
            for r, dirs, files in os.walk(dataset_path):
                if os.path.basename(r).lower() == split_name and dirs:
                    existing_splits[split_name] = r
                    break

        if len(existing_splits) == 3:
            print("📦 Found official train/val/test split folders.")
            for split_name, src_root in existing_splits.items():
                dst_root = os.path.join(MINI_SPLIT_ROOT, split_name)
                if not os.path.exists(dst_root):
                    shutil.copytree(src_root, dst_root)
            print("✅ miniImageNet Data Structure Ready! (official 64/16/20 split)")
            return

        # الحالة 2: مجلد واحد فيه كل الأصناف (100 مجلد) → نعمل تقسيم يدوي 64/16/20
        images_root = _find_class_dirs(dataset_path)
        if images_root is None:
            raise RuntimeError("Could not locate miniImageNet class directories in download.")

        print(f"📁 Found images root: {images_root}")
        classes = sorted([d for d in os.listdir(images_root)
                           if os.path.isdir(os.path.join(images_root, d))])
        print(f"📦 Found {len(classes)} classes.")

        rng = random.Random(42)  # seed ثابت ومستقل عن حالة random العامة —
                                  # عشان الـ split يتكرر بنفس الشكل بين
                                  # الجلسات، ويطابق باقي سكريباتنا (Kaggle/Colab).
        rng.shuffle(classes)
        splits = {
            "train": classes[:64],
            "val":   classes[64:80],
            "test":  classes[80:100],
        }

        for split_name, split_classes in splits.items():
            os.makedirs(os.path.join(MINI_SPLIT_ROOT, split_name), exist_ok=True)
            for c in split_classes:
                src = os.path.join(images_root, c)
                dst = os.path.join(MINI_SPLIT_ROOT, split_name, c)
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
        print("✅ miniImageNet Data Structure Ready! (64/16/20 split)")
    except Exception as e:
        print(f"❌ Error downloading/preparing miniImageNet: {e}")


with StageTimer("Stage 0: Data Preparation (miniImageNet)"):
    prepare_miniimagenet()

# ============================================
# 4. PROTOTYPE (Plain ProtoNet — no learnable norm, no gamma)
# ============================================

def compute_prototype(z_support):
    """Class prototype = arithmetic mean of raw support embeddings."""
    return z_support.mean(dim=0)

# ============================================
# 5. MODEL DEFINITION
# ============================================

def get_mobilenet_finetuned(num_classes=64, supervised=True):
    print(f"🔄 Loading MobileNetV3-SMALL (num_classes={num_classes})...")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model   = models.mobilenet_v3_small(weights=weights)

    if supervised:
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        model.classifier = nn.Identity()

    return model


def build_protonet_for_inference(checkpoint_path, num_classes=64):
    base_model = get_mobilenet_finetuned(num_classes=num_classes, supervised=True)
    base_model.classifier = nn.Identity()
    model = MobileProtoNet(base_model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    return model


class MobileProtoNet(nn.Module):
    """
    Plain ProtoNet: prototype = mean of support embeddings,
    logits = -squared_euclidean_distance(query, prototype).
    لا يوجد LearnableNorm ولا gamma (بدون أي معاملات إضافية قابلة للتعلم
    خارج الـ backbone نفسه).
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, support_x, query_x, n_way, k_shot):
        x = torch.cat([support_x, query_x], dim=0)
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        z = torch.flatten(x, 1)

        z_support = z[:support_x.size(0)]
        z_query   = z[support_x.size(0):]

        z_proto = torch.stack([
            compute_prototype(z_support[i * k_shot:(i + 1) * k_shot])
            for i in range(n_way)
        ])

        # ProtoNet القياسي يستخدم مربع المسافة الإقليدية
        dists = torch.cdist(z_query, z_proto) ** 2
        logits = -dists
        return logits

# ============================================
# 6. DATA LOADERS (parametrized by input_size)
# ============================================
# نفس منطق الـ transforms بالظبط زي نسخة Kaggle/Colab: كل الدقات (بما فيها
# 84×84) بتعدي على مسار resize→crop بنسبة هامش ثابتة (~12.5%)، عشان شدة
# augmentation تفضل متساوية بين كل الدقات وتبقى المقارنة عادلة.

def get_resize_dim(crop_dim):
    return round(crop_dim * 256 / 224)


def build_transforms(input_size):
    resize_dim = get_resize_dim(input_size)

    train_transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


MINI_ROOT = MINI_SPLIT_ROOT

# batch size أقل عند الدقات الكبيرة عشان نتجنب overflow في VRAM — نفس
# القاموس المستخدم في نسخة Kaggle/Colab، بدل batch_size=32 ثابت.
BATCH_SIZE_BY_RESOLUTION = {84: 64, 112: 48, 168: 32, 224: 32, 256: 16}


class EpisodicSampler:
    def __init__(self, root, transform):
        self.root      = root
        self.transform = transform
        self.classes   = sorted(os.listdir(root))
        self.data      = {}
        for c in self.classes:
            cls_path = os.path.join(root, c)
            self.data[c] = [
                os.path.join(cls_path, img)
                for img in os.listdir(cls_path)
                if img.lower().endswith(('jpg', 'jpeg', 'png'))
            ]

    def get_episode(self, n_way, k_shot, q_query):
        selected_classes = random.sample(self.classes, n_way)
        support_imgs, query_imgs, query_labels = [], [], []

        for i, cls in enumerate(selected_classes):
            available_imgs = self.data[cls]
            needed = k_shot + q_query
            if len(available_imgs) < needed:
                sampled_paths = random.choices(available_imgs, k=needed)
            else:
                sampled_paths = random.sample(available_imgs, needed)

            for path in sampled_paths[:k_shot]:
                img = Image.open(path).convert('RGB')
                support_imgs.append(self.transform(img))

            for path in sampled_paths[k_shot:]:
                img = Image.open(path).convert('RGB')
                query_imgs.append(self.transform(img))
                query_labels.append(i)

        return (torch.stack(support_imgs),
                torch.stack(query_imgs),
                torch.tensor(query_labels))

# ============================================
# 6b. MATCHED-EPISODE EVALUATION (fixed episode set)
# ============================================
# نولّد تعريفات الـ episodes (أصناف + مسارات صور الدعم/الاستعلام فقط، بدون
# تحميل الصور) مرة واحدة باستخدام RNG محلي منفصل عن الـ random العام، حتى
# لا تتأثر بأي seeding آخر في الكود. بعدين نفس التعريفات بتتحمّل وبتتحوّل
# بـ transform كل دقة صورة على حدة عند التقييم، فتبقى المقارنة "matched"
# تمامًا: نفس الأصناف ونفس الصور بالظبط لكل دقة.

def generate_fixed_episodes(root, n_way, k_shot, q_query, n_episodes, seed):
    rng = random.Random(seed)  # RNG محلي مستقل — لا يمس حالة random العامة
    classes = sorted(os.listdir(root))
    data = {}
    for c in classes:
        cls_path = os.path.join(root, c)
        data[c] = [
            os.path.join(cls_path, img)
            for img in os.listdir(cls_path)
            if img.lower().endswith(('jpg', 'jpeg', 'png'))
        ]

    episodes = []
    for _ in range(n_episodes):
        selected_classes = rng.sample(classes, n_way)
        support_paths, query_paths, query_labels = [], [], []

        for i, cls in enumerate(selected_classes):
            available_imgs = data[cls]
            needed = k_shot + q_query
            if len(available_imgs) < needed:
                sampled_paths = rng.choices(available_imgs, k=needed)
            else:
                sampled_paths = rng.sample(available_imgs, needed)

            support_paths.extend(sampled_paths[:k_shot])
            query_paths.extend(sampled_paths[k_shot:])
            query_labels.extend([i] * q_query)

        episodes.append({
            "classes": selected_classes,
            "support_paths": support_paths,
            "query_paths": query_paths,
            "query_labels": query_labels,
        })

    return episodes


def _load_episode_tensors(episode_def, transform, n_way, k_shot):
    support_imgs = [transform(Image.open(p).convert('RGB'))
                     for p in episode_def["support_paths"]]
    query_imgs   = [transform(Image.open(p).convert('RGB'))
                     for p in episode_def["query_paths"]]
    query_labels = torch.tensor(episode_def["query_labels"])
    return torch.stack(support_imgs), torch.stack(query_imgs), query_labels


@torch.no_grad()
def evaluate_fixed_episodes(model, episode_defs, transform, n_way, k_shot, desc=""):
    """يقيّم موديل على مجموعة episodes ثابتة (paths محددة سلفًا) ويرجّع
    مصفوفة دقة لكل episode على حدة (بدون تجميع)، عشان تُستخدم في اختبار
    paired لاحقًا."""
    model.eval()
    accs = []
    for ep_def in tqdm(episode_defs, desc=f"Matched-Eval {desc}", leave=False):
        sx, qx, qy = _load_episode_tensors(ep_def, transform, n_way, k_shot)
        sx, qx, qy = sx.to(device), qx.to(device), qy.to(device)
        logits = model(sx, qx, n_way, k_shot)
        preds  = logits.argmax(dim=1)
        accs.append((preds == qy).float().mean().item())
    return np.array(accs)


def paired_resolution_test(acc_84, acc_224, label=""):
    """اختبار paired بين نفس الـ episodes بدقتين مختلفتين.
    Δ_i = Acc224_i - Acc84_i ، H0: E[Δ] = 0."""
    assert len(acc_84) == len(acc_224), "عدد الـ episodes لازم يتطابق للمقارنة الـ paired"
    delta = acc_224 - acc_84
    mean_delta = delta.mean()
    std_delta  = delta.std(ddof=1)
    n = len(delta)
    se = std_delta / np.sqrt(n)
    ci95 = 1.96 * se

    t_stat, t_pval = scipy_stats.ttest_rel(acc_224, acc_84)
    try:
        w_stat, w_pval = scipy_stats.wilcoxon(acc_224, acc_84)
    except ValueError:
        # Wilcoxon بيفشل لو كل الفروق = صفر
        w_stat, w_pval = float('nan'), float('nan')

    print(f"\n📈 Paired 84x84 vs 224x224 — {label}")
    print(f"   n episodes        : {n}")
    print(f"   Mean Acc @84       : {acc_84.mean():.2%}")
    print(f"   Mean Acc @224      : {acc_224.mean():.2%}")
    print(f"   Mean Δ (224-84)    : {mean_delta:+.2%}  (95% CI ±{ci95:.2%})")
    print(f"   Paired t-test      : t={t_stat:.4f}, p={t_pval:.6f}")
    print(f"   Wilcoxon signed-rank: W={w_stat:.4f}, p={w_pval:.6f}")

    return {
        "n": n,
        "mean_delta": mean_delta,
        "ci95_delta": ci95,
        "t_stat": t_stat,
        "t_pval": t_pval,
        "w_stat": w_stat,
        "w_pval": w_pval,
        "acc_84_mean": acc_84.mean(),
        "acc_224_mean": acc_224.mean(),
    }

# ============================================
# 7. EXECUTION STAGES
# ============================================

def stage1_finetuning(train_loader_simple, epochs=10, num_classes=64):
    print(f"\n🦾 STAGE 1: Fine-tuning MobileNet on miniImageNet for {epochs} epochs...")

    model     = get_mobilenet_finetuned(num_classes=num_classes, supervised=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                 nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader_simple, desc=f"Fine-tune Ep {epoch+1}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            pbar.set_postfix({'Acc': f"{correct/total:.1%}"})

        scheduler.step()
        print(f"   Ep {epoch+1}: Loss {total_loss/len(train_loader_simple):.4f} | "
              f"Train Acc {correct/total:.2%}")

    print("✅ Fine-tuning Complete.")
    return model


def stage2_metatrain(pretrained_state_dict, train_sampler, val_sampler,
                      n_way=5, k_shot=1, episodes=2000, num_classes=64,
                      ckpt_path="best_mobile_proto_miniimagenet.pth"):
    print(f"\n🧠 STAGE 2: Meta-Training (ProtoNet) on miniImageNet "
          f"{n_way}-way {k_shot}-shot...")

    base_model = get_mobilenet_finetuned(num_classes=num_classes, supervised=True)
    base_model.load_state_dict(pretrained_state_dict)
    base_model.classifier = nn.Identity()

    model = MobileProtoNet(base_model).to(device)

    # لا يوجد learnable_norm أو gamma الآن، لذلك مجموعة معاملات واحدة فقط للـ backbone
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience     = 0

    for ep in tqdm(range(episodes), desc="Meta-Train"):
        sx, qx, qy = train_sampler.get_episode(n_way, k_shot, q_query=15)
        sx, qx, qy = sx.to(device), qx.to(device), qy.to(device)

        model.train()
        optimizer.zero_grad()
        logits = model(sx, qx, n_way, k_shot)
        loss   = criterion(logits, qy)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 200 == 0:
            val_acc, _ = evaluate(model, val_sampler, n_way, k_shot,
                                   episodes=200, silent=True)
            print(f"\n   Ep {ep+1}: Loss {loss.item():.4f} | Val Acc: {val_acc:.2%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                patience = 0
            else:
                patience += 1
                if patience > 10:
                    print("⏹️ Early stopping triggered.")
                    break

    return model


@torch.no_grad()
def evaluate(model, sampler, n_way, k_shot, episodes=500, silent=False):
    model.eval()
    accs     = []
    iterator = range(episodes) if silent else tqdm(range(episodes), desc="Evaluating")

    for _ in iterator:
        sx, qx, qy = sampler.get_episode(n_way, k_shot, q_query=15)
        sx, qx, qy = sx.to(device), qx.to(device), qy.to(device)

        logits = model(sx, qx, n_way, k_shot)
        preds  = logits.argmax(dim=1)
        accs.append((preds == qy).float().mean().item())

    mean_acc = np.mean(accs)
    ci95     = 1.96 * np.std(accs) / np.sqrt(episodes)

    if not silent:
        print(f"📊 Result: {mean_acc:.2%} ± {ci95:.2%}")

    return mean_acc, ci95

# ============================================
# 8. RUN ONE FULL EXPERIMENT (given input_size + seed)
# ============================================

NUM_TRAIN_CLASSES = 64  # miniImageNet train split size

def run_experiment(input_size, seed, epochs=10, episodes=2000):
    print("\n" + "#"*60)
    print(f"### RUNNING EXPERIMENT: input_size={input_size}x{input_size} | seed={seed}")
    print("#"*60)

    set_seed(seed)

    train_transform, test_transform = build_transforms(input_size)

    train_dataset_simple = datasets.ImageFolder(
        root=os.path.join(MINI_ROOT, "train"),
        transform=train_transform
    )
    batch_size = BATCH_SIZE_BY_RESOLUTION.get(input_size, 32)
    train_loader_simple = DataLoader(
        train_dataset_simple,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print(f"   Batch size: {batch_size}")

    train_sampler = EpisodicSampler(os.path.join(MINI_ROOT, "train"), test_transform)
    val_sampler   = EpisodicSampler(os.path.join(MINI_ROOT, "val"),   test_transform)
    test_sampler  = EpisodicSampler(os.path.join(MINI_ROOT, "test"),  test_transform)

    tag = f"{input_size}px_seed{seed}"

    # ── Stage 1 ──────────────────────────────
    backbone_ckpt = os.path.join(WORK_ROOT, f"backbone_pretrained_miniimagenet_{tag}.pth")
    if not os.path.exists(backbone_ckpt):
        with StageTimer(f"Stage 1: Fine-Tuning ({tag})"):
            ft_model = stage1_finetuning(train_loader_simple, epochs=epochs,
                                          num_classes=NUM_TRAIN_CLASSES)
            torch.save(ft_model.state_dict(), backbone_ckpt)
    else:
        print(f"✅ Pre-trained backbone found for {tag}. Skipping Stage 1.")
    pretrained_state_dict = torch.load(backbone_ckpt, map_location=device)

    results = {}

    for k_shot in [1, 5]:
        print("\n" + "="*40)
        print(f"🎯 TASK: 5-way {k_shot}-shot ({tag})")
        print("="*40)

        ckpt_path = os.path.join(
            WORK_ROOT, f"best_mobile_proto_miniimagenet_{tag}_{k_shot}shot.pth"
        )

        with StageTimer(f"Stage 2: Meta-Training {k_shot}-shot ({tag})"):
            stage2_metatrain(pretrained_state_dict, train_sampler, val_sampler,
                              n_way=5, k_shot=k_shot, episodes=episodes,
                              num_classes=NUM_TRAIN_CLASSES, ckpt_path=ckpt_path)

        with StageTimer(f"Stage 3: Test Evaluation {k_shot}-shot ({tag})"):
            best_model = build_protonet_for_inference(
                checkpoint_path=ckpt_path, num_classes=NUM_TRAIN_CLASSES
            )
            mean_acc, ci95 = evaluate(best_model, test_sampler, n_way=5, k_shot=k_shot)
            results[k_shot] = {"mean_acc": mean_acc, "ci95": ci95, "ckpt_path": ckpt_path}

        if k_shot == 5:
            print("\n" + "="*40)
            print(f"📐 MODEL SIZE & COMPUTE REPORT ({tag})")
            print("="*40)
            report_full_model_stats(best_model, input_size=input_size)

    return results

# ============================================
# 8b. RESUME / PROGRESS TRACKING
# ============================================
# ملف تقدّم بيتحدّث بعد كل (input_size, seed, k_shot) يخلص — لو التجربة
# اتقطعت (انقطع النت، قفلت الجلسة، ...) وشغّلت الكود تاني، هيقرأ الملف ده
# ويكمّل من عند آخر حاجة خلصها، من غير ما يعيد أي تدريب أو تقييم اتعمل قبل
# كده. الحفظ atomic (كتابة لملف مؤقت ثم rename) عشان لو الجلسة اتقفلت في
# نص الكتابة، الملف الأصلي يفضل سليم.

PROGRESS_FILE = os.path.join(WORK_ROOT, "progress.json")


def load_progress():
    """يرجّع (all_results, matched_accs) من ملف التقدّم لو موجود، وإلا
    قواميس فاضية. all_results بنفس شكل المتغيّر all_results في MAIN.
    matched_accs بنفس شكل matched_accs (مصفوفات per-episode لكل
    k_shot/seed/input_size، مستخدمة في الـ Matched-Episode Evaluation)."""
    if not os.path.exists(PROGRESS_FILE):
        return {}, {}
    try:
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  ملف التقدّم {PROGRESS_FILE} تالف ({e}) — هنبدأ من الصفر.")
        return {}, {}

    all_results = {}
    for res_str, seeds in data.get("results", {}).items():
        res = int(res_str)
        all_results[res] = {}
        for seed_str, kshots in seeds.items():
            seed = int(seed_str)
            all_results[res][seed] = {}
            for k_str, v in kshots.items():
                all_results[res][seed][int(k_str)] = {
                    "mean_acc": v["mean_acc"],
                    "ci95": v["ci95"],
                    "ckpt_path": v["ckpt_path"],
                }

    matched_accs = {}
    for k_str, seeds in data.get("matched", {}).items():
        k_shot = int(k_str)
        matched_accs[k_shot] = {}
        for seed_str, res_dict in seeds.items():
            seed = int(seed_str)
            matched_accs[k_shot][seed] = {
                84:  np.array(res_dict["84"]),
                224: np.array(res_dict["224"]),
            }

    if all_results or matched_accs:
        print(f"\n♻️  لقينا تقدّم سابق في {PROGRESS_FILE}:")
        for res, seeds in all_results.items():
            print(f"    {res}px: seeds خلّصت = {sorted(seeds.keys())}")
        for k_shot, seeds in matched_accs.items():
            print(f"    Matched-Eval {k_shot}-shot: seeds خلّصت = {sorted(seeds.keys())}")

    return all_results, matched_accs


def save_progress(all_results, matched_accs):
    """يحفظ التقدّم الحالي بطريقة atomic (كتابة لملف مؤقت ثم rename)."""
    report = {
        "results": {
            str(res): {
                str(seed): {
                    str(k_shot): {
                        "mean_acc": float(v["mean_acc"]),
                        "ci95": float(v["ci95"]),
                        "ckpt_path": v["ckpt_path"],
                    }
                    for k_shot, v in kshots.items()
                }
                for seed, kshots in seeds.items()
            }
            for res, seeds in all_results.items()
        },
        "matched": {
            str(k_shot): {
                str(seed): {
                    "84":  res_dict[84].tolist(),
                    "224": res_dict[224].tolist(),
                }
                for seed, res_dict in seeds.items()
            }
            for k_shot, seeds in matched_accs.items()
        },
    }
    tmp_path = PROGRESS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, PROGRESS_FILE)  # rename ذرّي — إما ينجح كامل أو مايحصلش خالص

# ============================================
# 9. MAIN — LOOP OVER INPUT SIZES × 3 SEEDS
# ============================================

if __name__ == "__main__":
    INPUT_SIZES = [84, 224]
    SEEDS       = [1, 2, 3]
    N_WAY       = 5
    Q_QUERY     = 15
    N_MATCHED_EPISODES = 2000   # عدد الـ episodes الثابتة للمقارنة الـ paired
    MATCHED_EPISODES_SEED = 999  # seed مستقل عن seeds التدريب — نفس الـ episodes تُستخدم مع كل الـ seeds

    all_results, matched_accs = load_progress()

    for input_size in INPUT_SIZES:
        all_results.setdefault(input_size, {})
        for seed in SEEDS:
            already_done = (
                seed in all_results[input_size]
                and 1 in all_results[input_size][seed]
                and 5 in all_results[input_size][seed]
            )
            if already_done:
                print(f"✅ {input_size}px seed={seed} موجود في التقدّم المحفوظ — تخطّي.")
                continue

            res = run_experiment(input_size, seed, epochs=10, episodes=2000)
            all_results[input_size][seed] = res
            save_progress(all_results, matched_accs)  # حفظ فوري بعد كل seed

    # ── ملخص النتائج عبر الـ 3 seeds ──────────
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY — MEAN ± STD OVER 3 SEEDS")
    print("="*60)
    for input_size in INPUT_SIZES:
        for k_shot in [1, 5]:
            accs = [all_results[input_size][s][k_shot]["mean_acc"] for s in SEEDS]
            print(f"  {input_size}x{input_size} | {k_shot}-shot: "
                  f"{np.mean(accs):.2%} ± {np.std(accs):.2%}  "
                  f"(runs: {[f'{a:.2%}' for a in accs]})")

    # ============================================
    # MATCHED-EPISODE EVALUATION (84x84 vs 224x224)
    # ============================================
    # نولّد مجموعة episodes ثابتة (نفس الأصناف ونفس صور الدعم/الاستعلام)
    # مرة واحدة لكل k-shot، مستقلة عن seeds التدريب، ونعيد استخدامها مع
    # كل الـ seeds. المقارنة بين 84x84 و224x224 على نفس الـ episodes
    # بالظبط تحوّلها لمقارنة paired أقوى إحصائيًا من مقارنة متوسطين من
    # مجموعتين مختلفتين من episodes عشوائية.
    print("\n" + "="*60)
    print("📈 MATCHED-EPISODE EVALUATION (paired 84x84 vs 224x224)")
    print("="*60)

    with StageTimer("Stage 4: Matched-Episode Generation"):
        fixed_episodes = {
            k_shot: generate_fixed_episodes(
                root=os.path.join(MINI_ROOT, "test"),
                n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY,
                n_episodes=N_MATCHED_EPISODES, seed=MATCHED_EPISODES_SEED
            )
            for k_shot in [1, 5]
        }

    _, test_transform_84  = build_transforms(84)
    _, test_transform_224 = build_transforms(224)

    with StageTimer("Stage 5: Matched-Episode Evaluation"):
        for k_shot in [1, 5]:
            matched_accs.setdefault(k_shot, {})
            eps = fixed_episodes[k_shot]

            for seed in SEEDS:
                if seed in matched_accs[k_shot]:
                    print(f"✅ Matched-Eval {k_shot}-shot seed={seed} "
                          f"موجود في التقدّم المحفوظ — تخطّي.")
                    continue

                ckpt_84  = all_results[84][seed][k_shot]["ckpt_path"]
                ckpt_224 = all_results[224][seed][k_shot]["ckpt_path"]

                model_84  = build_protonet_for_inference(ckpt_84,  num_classes=NUM_TRAIN_CLASSES)
                model_224 = build_protonet_for_inference(ckpt_224, num_classes=NUM_TRAIN_CLASSES)

                acc_84  = evaluate_fixed_episodes(model_84,  eps, test_transform_84,
                                                   N_WAY, k_shot, desc=f"84px seed{seed} {k_shot}shot")
                acc_224 = evaluate_fixed_episodes(model_224, eps, test_transform_224,
                                                   N_WAY, k_shot, desc=f"224px seed{seed} {k_shot}shot")

                matched_accs[k_shot][seed] = {84: acc_84, 224: acc_224}
                save_progress(all_results, matched_accs)  # حفظ فوري بعد كل seed

                paired_resolution_test(acc_84, acc_224,
                                        label=f"{k_shot}-shot, seed={seed}")

            # ── تجميع كل الـ seeds مع بعض (نفس الـ episodes الثابتة مُعادة الاستخدام
            #    مع كل seed، فالـ pairing يفضل صحيح لكل عنصر) ──
            pooled_84  = np.concatenate([matched_accs[k_shot][s][84]  for s in SEEDS])
            pooled_224 = np.concatenate([matched_accs[k_shot][s][224] for s in SEEDS])
            paired_resolution_test(pooled_84, pooled_224,
                                    label=f"{k_shot}-shot, POOLED across {len(SEEDS)} seeds "
                                          f"({N_MATCHED_EPISODES} episodes × {len(SEEDS)} seeds)")

    print_stage_time_summary()
    print(f"\n💾 التقدّم والنتائج محفوظة في: {PROGRESS_FILE}")
