# sort_fer2013.py
import os, shutil

# FER2013 extracted directly into raw folder
FER_ROOT    = r"C:\Users\way2s\pain_detection_edge\dataset\raw"
OUTPUT_ROOT = r"C:\Users\way2s\pain_detection_edge\dataset\raw"

# Pain emotions — share same facial muscles as real pain
PAIN_EMOTIONS    = ["angry", "disgust", "fear", "sad"]
NO_PAIN_EMOTIONS = ["happy", "neutral", "surprise"]

# Create fresh pain/no_pain folders
for folder in ["pain", "no_pain"]:
    path = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {folder}/")

pain_count    = 0
no_pain_count = 0

for split in ["train", "test"]:
    split_path = os.path.join(FER_ROOT, split)
    if not os.path.exists(split_path):
        print(f"Skipping {split} - not found")
        continue

    print(f"\nProcessing {split}/...")

    for emotion in sorted(os.listdir(split_path)):
        emotion_path = os.path.join(split_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        if emotion.lower() in PAIN_EMOTIONS:
            dst_folder = os.path.join(OUTPUT_ROOT, "pain")
            label      = "pain"
        elif emotion.lower() in NO_PAIN_EMOTIONS:
            dst_folder = os.path.join(OUTPUT_ROOT, "no_pain")
            label      = "no_pain"
        else:
            print(f"  Skipping unknown: {emotion}")
            continue

        count = 0
        for fname in os.listdir(emotion_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(emotion_path, fname)
                dst = os.path.join(
                    dst_folder,
                    f"{split}_{emotion}_{fname}"
                )
                shutil.copy2(src, dst)
                count += 1
                if label == "pain":
                    pain_count += 1
                else:
                    no_pain_count += 1

        print(f"  {split}/{emotion} -> {label}/ : {count} images")

print(f"\n{'='*45}")
print(f"Sorting COMPLETE!")
print(f"  Pain    : {pain_count}")
print(f"  No Pain : {no_pain_count}")
print(f"  Total   : {pain_count + no_pain_count}")
print(f"{'='*45}")
print(f"\nSaved to: {OUTPUT_ROOT}")