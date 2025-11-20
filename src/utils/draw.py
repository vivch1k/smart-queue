import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO

matplotlib.use("Agg")

def draw_ru_text(frame, position, size, color, text):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("src/assets/fonts/arialmt.ttf", size=size)
    draw.text(position, text, font=font, fill=color)
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return frame


def draw_gradient_indicator(frame, value, position=(50, 50), size=(300, 30), thickness=2, label="Касса", 
                             font_size=15, color="black", text=True):
    x, y = position
    w, h = size
    value = np.clip(value, 0, 1)  

    gradient = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        ratio = i / w
        if ratio < 0.5:
            r = int(255 * (ratio / 0.5))      
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (1 - (ratio - 0.5) / 0.5)) 
            b = 0
        gradient[:, i] = (b, g, r)

    filled = int(w * value)
    bar = gradient.copy()
    mask = np.zeros_like(bar, dtype=np.uint8)
    mask[:, :filled] = 1
    filled_bar = cv2.bitwise_and(bar, bar, mask=mask[:, :, 0])

    overlay = frame.copy()
    overlay[y:y+h, x:x+w] = filled_bar
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), thickness)
    if text:
        frame = draw_ru_text(frame, (x+6, y-20), size=font_size, color=color, text=label)
        

    return frame


def draw_cash_info(frame, zone_idx, indicator_x, indicator_y, load, count_in_zone, service_time):
    overlay = frame.copy()
    img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.rounded_rectangle(
        [(indicator_x - 5, indicator_y - 25), (indicator_x + 75, indicator_y + 45)],
        radius=8,
        fill=(0, 0, 0),
        outline=(0, 0, 0),
        width=2
    )
    overlay = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    frame = draw_gradient_indicator(frame=frame,
                                    value=load,
                                    position=(indicator_x, indicator_y),
                                    size=(70, 16),
                                    label=f"Касса {zone_idx+1}",
                                    font_size=17,
                                    color="white")
    
    info_text = f"{count_in_zone}чел | {service_time:.0f}с"
    frame = draw_ru_text(frame, (indicator_x+10, indicator_y + 25), size=14, color="white", text=info_text)

    return frame


def draw_global_panel(frame, max_load, max_zone_idx):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    frame = draw_gradient_indicator(frame=frame,
                                    value=max_load,
                                    position=(70, h - 60),
                                    size=(300, 30),
                                    label=f"Макс. загрузка {max_load*100:.0f}% (касса {max_zone_idx+1})",
                                    font_size=19,
                                    color="white")

    if max_load >= 0.6:
        frame = draw_ru_text(frame, (520, h - 60), size=30, color="red", text="Необходимо открыть кассу")
    

    return frame


def draw_curves(time, load, size, fig_label, ylim, max_val=100):
    n = len(load)
    fig, ax = plt.subplots(len(load), 1, figsize=size)
    
    fig.suptitle(fig_label, fontsize=14)

    tm = time[-max_val:]
    if n > 0:
        for i in range(n):
            load_zone = load[i][-max_val:]
            ax[i].plot(tm, load_zone, c="blue")
            ax[i].set_title(f"Касса {i+1}", fontdict={"fontsize": 12}, pad=1)
            ax[i].set_ylim(*ylim)
            ax[i].grid(alpha=0.3)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)

    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    return img

class CurvesPlotUpdate:
    def __init__(self, fps):
        self.fps = fps

        self.curves_load = None
        self.curves_people = None
        self.curves_service = None

    def update(self, curr_frame, frames, load_zones, people_zones, service_zones):

        if curr_frame % self.fps == 0:
            self.curves_load = draw_curves(frames, load_zones, (3, 7), "Загруженость", (0, 1), 120)
            self.curves_people = draw_curves(frames, people_zones, (3, 7), "Кол-во людей", (0, 10), 120)
            self.curves_service = draw_curves(frames, service_zones, (3, 7), "Время обслуживания", (0, 120), 120)

        return self.curves_load, self.curves_people, self.curves_service