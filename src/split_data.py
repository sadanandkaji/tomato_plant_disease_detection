import os, shutil, random
random.seed(42)
SRC, DST = "data/raw", "data/split"

for cls in sorted(os.listdir(SRC)):
    files = sorted(os.listdir(os.path.join(SRC, cls)))
    random.shuffle(files)
    n = len(files)
    cuts = {"train": files[:int(.70*n)],
            "val":   files[int(.70*n):int(.85*n)],
            "test":  files[int(.85*n):]}
    for split, fl in cuts.items():
        os.makedirs(f"{DST}/{split}/{cls}", exist_ok=True)
        for f in fl:
            shutil.copy(f"{SRC}/{cls}/{f}", f"{DST}/{split}/{cls}/{f}")
    print(cls, {k: len(v) for k, v in cuts.items()})
