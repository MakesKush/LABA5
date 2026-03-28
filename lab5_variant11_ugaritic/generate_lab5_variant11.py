from pathlib import Path
import csv
import math
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / 'source_symbols'
RESULTS_DIR = ROOT / 'results'
FONT_PATH = '/usr/share/fonts/truetype/noto/NotoSansUgaritic-Regular.ttf'
FONT_SIZE = 96
CODEPOINTS = list(range(0x10380, 0x1039E))  # 30 букв угаритского алфавита



def render_symbol(ch, font):
    img = Image.new('L', (220, 220), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    x = (img.width - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (img.height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)

    arr = np.array(img)
    mask = arr < 200
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    cropped = img.crop((left, top, right + 1, bottom + 1))

    padded = Image.new('L', (cropped.width + 4, cropped.height + 4), 255)
    padded.paste(cropped, (2, 2))
    return padded



def to_binary(img):
    return (np.array(img) < 200).astype(np.uint8)



def compute_features(binary):
    h, w = binary.shape
    total_weight = int(binary.sum())

    mid_x = w // 2
    mid_y = h // 2
    quarters = [
        binary[:mid_y, :mid_x],
        binary[:mid_y, mid_x:],
        binary[mid_y:, :mid_x],
        binary[mid_y:, mid_x:],
    ]
    q_weights = [int(q.sum()) for q in quarters]
    q_areas = [int(q.shape[0] * q.shape[1]) for q in quarters]
    q_densities = [q_weights[i] / q_areas[i] if q_areas[i] else 0.0 for i in range(4)]

    ys, xs = np.indices(binary.shape)
    m00 = float(total_weight)
    if m00 == 0:
        xc = yc = xc_norm = yc_norm = mu20 = mu02 = 0.0
    else:
        m10 = float((xs * binary).sum())
        m01 = float((ys * binary).sum())
        xc = m10 / m00
        yc = m01 / m00
        xc_norm = xc / (w - 1) if w > 1 else 0.0
        yc_norm = yc / (h - 1) if h > 1 else 0.0
        mu20 = float((((xs - xc) ** 2) * binary).sum())
        mu02 = float((((ys - yc) ** 2) * binary).sum())

    Iy = mu20
    Ix = mu02
    area = w * h

    return {
        'width': w,
        'height': h,
        'weight': total_weight,
        'q1_weight': q_weights[0],
        'q2_weight': q_weights[1],
        'q3_weight': q_weights[2],
        'q4_weight': q_weights[3],
        'q1_density': q_densities[0],
        'q2_density': q_densities[1],
        'q3_density': q_densities[2],
        'q4_density': q_densities[3],
        'xc': xc,
        'yc': yc,
        'xc_norm': xc_norm,
        'yc_norm': yc_norm,
        'Ix': Ix,
        'Iy': Iy,
        'Ix_norm': Ix / area if area else 0.0,
        'Iy_norm': Iy / area if area else 0.0,
        'profile_x': binary.sum(axis=0).astype(int),
        'profile_y': binary.sum(axis=1).astype(int),
    }



def save_profile(values, out_path, title, x_label, y_label):
    plt.figure(figsize=(8, 3.8))
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if len(values) > 25:
        step = max(1, len(values) // 10)
        plt.xticks(np.arange(0, len(values), step))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



def save_gallery(image_paths):
    images = [Image.open(p) for p in image_paths]
    thumb_w = max(img.width for img in images)
    thumb_h = max(img.height for img in images)

    cols = 5
    rows = math.ceil(len(images) / cols)
    margin = 14
    label_h = 28
    cell_w = thumb_w + margin * 2
    cell_h = thumb_h + label_h + margin * 2

    canvas = Image.new('L', (cols * cell_w, rows * cell_h), 255)
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * cell_w + (cell_w - img.width) // 2
        y = row * cell_h + margin
        canvas.paste(img, (x, y))

        cp = CODEPOINTS[i]
        label = f'{i + 1:02d} U+{cp:05X}'
        bbox = draw.textbbox((0, 0), label, font=label_font)
        tx = col * cell_w + (cell_w - (bbox[2] - bbox[0])) // 2
        ty = row * cell_h + thumb_h + margin + 6
        draw.text((tx, ty), label, fill=0, font=label_font)

    canvas.save(RESULTS_DIR / '00_alphabet_gallery.png')



def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    rows = []
    source_paths = []

    for idx, cp in enumerate(CODEPOINTS, start=1):
        ch = chr(cp)
        name = unicodedata.name(ch, f'U+{cp:05X}')
        symbol_img = render_symbol(ch, font)

        source_path = SOURCE_DIR / f'{idx:02d}_u{cp:05X}.png'.lower()
        symbol_img.save(source_path)
        source_paths.append(source_path)

        binary = to_binary(symbol_img)
        feats = compute_features(binary)

        out_dir = RESULTS_DIR / f'{idx:02d}_u{cp:05X}'.lower()
        out_dir.mkdir(exist_ok=True)
        symbol_img.save(out_dir / '00_symbol.png')
        save_profile(feats['profile_x'], out_dir / '01_profile_x.png', f'Профиль X: U+{cp:05X}', 'x', 'PX(x)')
        save_profile(feats['profile_y'], out_dir / '02_profile_y.png', f'Профиль Y: U+{cp:05X}', 'y', 'PY(y)')

        row = {
            'id': idx,
            'codepoint': f'U+{cp:05X}',
            'symbol': ch,
            'name': name.replace('UGARITIC LETTER ', '').title(),
        }
        for key, value in feats.items():
            if isinstance(value, np.ndarray):
                continue
            row[key] = value
        rows.append(row)

    fieldnames = [
        'id', 'codepoint', 'symbol', 'name', 'width', 'height', 'weight',
        'q1_weight', 'q2_weight', 'q3_weight', 'q4_weight',
        'q1_density', 'q2_density', 'q3_density', 'q4_density',
        'xc', 'yc', 'xc_norm', 'yc_norm', 'Ix', 'Iy', 'Ix_norm', 'Iy_norm'
    ]
    with open(RESULTS_DIR / 'summary.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row[key]
                formatted[key] = f'{value:.6f}' if isinstance(value, float) else value
            writer.writerow(formatted)

    save_gallery(source_paths)
    print('Готово:', ROOT)


if __name__ == '__main__':
    main()
