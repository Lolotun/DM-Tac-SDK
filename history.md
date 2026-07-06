# Session history

## Sensor & SDK

- Сенсор **DM-Tac W2L** — оптико-тактильный: камера смотрит на гелевую поверхность, нейросеть перестраивает деформацию в данные.
- Пакет `dmrobotics` (Daimon) — закрытый, ядро зашифровано Pyarmor. Открыты только `__init__.py`, `utils.py`.
- Pipeline: сырое изображение → **FastFlowNet** → оптический поток (deformation/shear) → reconstruction nets → depth, force, distributed force.
- Три бекенда: `cpu` (ONNXRuntime), `cuda` (TensorRT), `Flux` (удалённый gRPC-сервер 192.168.127.10:50051).

## Что выдаёт API

| Метод | Что возвращает |
|---|---|
| `getRawImg()` / `getInferImg()` | `(fid, DMTacImage)` — сырой/обработанный кадр |
| `getDeformation2D()` | `(fid, HxWx2 float32)` — 2D поле смещений геля |
| `getShear()` | `(fid, HxWx2 float32)` — сдвиговые деформации |
| `getDepth()` | `(fid, HxW float32)` — карта вдавливания |
| `getForce()` | `[Fx, Fy, Fz, Mx, My, Mz]` — суммарный wrench |
| `getDistributeForce()` | распределённая сила по пикселям |
| `getContactArea()` | площадь/маска контакта |

Разрешение карт: **288×384** = 110 592 пикселей.

## Установка

```bash
pip install .          # CPU
pip install .[gpu]     # CUDA — нужен driver + CUDA Toolkit + cuDNN
dmrobotics trt build   # компилирует TensorRT engines (один раз, повторить после смены GPU/CUDA)
```

**GPU статус:** RTX 5090 присутствует, драйвер 595.71 (CUDA 13.2), но `libcudart.so` не найдена → `dmrobotics trt build` упал. Решение — `uv pip install nvidia-cuda-runtime-cu12` + `export DMSDK_CUDART_PATH=...`. Пока оставлено на CPU.

## Структура проекта (после реорганизации)

```
DM-Tac-SDK/
├── vendor/SDK_Publish_1.2.10/   # оригинальный SDK от Daimon, не трогать
│   ├── dmrobotics/              # пакет (lib/*.so — 562 МБ бинарей, gitignored)
│   ├── Daimon/                  # pyarmor runtime
│   └── setup.py
├── venv/                        # python 3.11.15 (uv), pip install -e ./vendor/SDK_Publish_1.2.10
├── scripts/
│   ├── live_view/               # main.py, main_mp.py, demo.py
│   ├── recording/               # get_force.py/.1.py/_new.py, tactile_video_manual_csv.py, save_img.py
│   ├── offline/                 # gen_feat_hdf5.py, parse_data.py, read.py, check_data.py, gen_trt.py
│   ├── slip_detection/          # slip_detection.py
│   └── standalone/              # tactile.py (без dmrobotics, чистый OpenCV)
├── data/                        # recordings/, recordings_with_water/, test.h5 (gitignored)
└── logs/                        # gitignored
```

## Данные

### Записанные сессии

- `data/recordings/` — 10 снэпшотов, сухой тест, 16:27 18.06.2026 (~20 ГБ)
- `data/recordings_with_water/` — 11 снэпшотов, тест с водой, 17:07 18.06.2026 (~12 ГБ)
- Видеофайлы `.mp4` не создались (enable_raw не сработал или writer упал молча)
- Каждый снэпшот = **все кадры с начала сеанса** (накопительно), не дельта

### Формат CSV

```
frame_id, time_s, fx, fy, fz, mx, my, mz,
depth_values,          # 110592 float32 через ";"  → reshape(288,384)
deformation_x_values,  # 110592 float32 через ";"
deformation_y_values   # 110592 float32 через ";"
```

### Конвертация CSV → NPZ (рекомендуется)

```python
def load_csv_session(path):
    df = pd.read_csv(path)
    N = len(df)
    force = df[["fx","fy","fz","mx","my","mz"]].to_numpy(dtype=np.float32)
    depth = np.vstack([np.fromstring(s, sep=";") for s in df["depth_values"]]).reshape(N, 288, 384)
    def_x = np.vstack([np.fromstring(s, sep=";") for s in df["deformation_x_values"]]).reshape(N, 288, 384)
    def_y = np.vstack([np.fromstring(s, sep=";") for s in df["deformation_y_values"]]).reshape(N, 288, 384)
    return df["time_s"].to_numpy(), force, depth, np.stack([def_x, def_y], axis=-1)

# сохранить:
time, force, depth, deform = load_csv_session("session.csv")
np.savez_compressed("session.npz",
    time=time, force=force, depth=depth, deformation=deform)

# загрузить потом:
data = np.load("session.npz")
depth_frame_42 = data["depth"][42]   # (288,384), без парсинга
```

## Баги найденные в скриптах

- `parse_data.py` — `def main()` без `if __name__ == "__main__": main()` → функция никогда не вызывалась, скрипт выдавал только CuPy warning и выходил.
- `parse_data.py` — `np.mean(prev_time - time[:-1])` → 0.0 из-за pandas index alignment: срезы сохраняют метки индексов, вычитание идёт label-by-label (row_i - row_i = 0). Плюс `numpy` не импортирован. Правильно: `df["time_s"].diff().mean()`.
- `read.py` — захардкожен путь `/home/physicalai/...` (другая машина).

## Полезные инструменты из utils.py

- `put_arrows_on_image(image, flow_hw2)` — рисует стрелки оптического потока (для визуализации deformation)
- `read_h5 / init_h5 / append_h5` — работа с HDF5 (только сырые кадры DMTacImage, не force/depth)

## .gitignore исключает

`venv/`, `vendor/.../build/`, `vendor/.../dmrobotics.egg-info/`, `vendor/.../dmrobotics/lib/`, `logs/*.log`, `data/`

---

## Что сделано (реорганизация и git)

### Среда (venv)
- Старый `venv` в `SDK_Publish_1.2.10/venv/` был нерабочим: `pyvenv.cfg` ссылался на `/home/physicalai/miniconda3/...` (другая машина).
- Пересобран через `uv` (python 3.11.15 из conda env `g1-kinematics`) в `DM-Tac-SDK/venv/`.
- Установка: `uv pip install -e ./vendor/SDK_Publish_1.2.10[gpu]` — GPU-режим через pip, но `dmrobotics trt build` упал.
- **GPU trt build**: `libcudart.so` не найдена ни в venv, ни в системе (только driver, Toolkit не установлен). Фикс: `uv pip install nvidia-cuda-runtime-cu12` + `export DMSDK_CUDART_PATH=...`. Пока **оставлено на CPU**.
- `uv` установлен в `/home/sa/snap/code/247/.local/bin/uv` (VSCode snap-окружение, не виден в обычном PATH).

### Структура репозитория
Реорганизация из плоского `SDK_Publish_1.2.10/` в:
```
vendor/SDK_Publish_1.2.10/   ← оригинал SDK нетронутый
scripts/{live_view,recording,offline,slip_detection,standalone}/
data/                        ← gitignored, 32 ГБ
venv/                        ← gitignored (internal .gitignore с *)
```

### Git
- `git init` сделан пользователем вручную, первый коммит "mesh with vibecoded files".
- Второй коммит `d546931`: реорганизация + обновлённый `.gitignore`.
- В коммит вошли: все скрипты, vendor python-код, Daimon/*.so (8 бинарей pyarmor runtime, ~5.6 МБ, необходимы для работы SDK), `depth.gif`/`depth_frame*.png` (визуализации depth-карт), `history.md`.
- Не вошли: `vendor/.../dmrobotics/lib/` (562 МБ), `vendor/.../build/` (571 МБ), `data/` (32 ГБ), `venv/`.
- Подключение remote к GitHub: `git remote add origin <URL>` — **ещё не сделано**, ждёт URL репозитория от пользователя.

### Скрипты пользователя (написаны самостоятельно)
- `scripts/offline/parse_data.py` — анализ CSV: средний интервал между кадрами, парсинг depth/deformation.
- `scripts/offline/csv_to_npz.py` — конвертер CSV → NPZ (написан пользователем).
- Визуализации `depth.gif`, `depth_frame.png`, `depth_frame_cv.png` — результаты работы с данными.

### Паттерн загрузки данных (итоговый)
```python
# правильный diff по времени (не через срезы — index alignment!)
df["time_s"].diff().mean()   # ~0.099 с между кадрами ≈ 10 FPS

# парсинг dense-полей из CSV
depth = np.vstack([np.fromstring(s, sep=";") for s in df["depth_values"]]).reshape(N, 288, 384)
deform = np.stack([...def_x..., ...def_y...], axis=-1)  # (N,288,384,2)

# рекомендуемый формат хранения — NPZ
np.savez_compressed("session.npz", time=..., force=..., depth=..., deformation=...)
data = np.load("session.npz")
data["depth"][frame_idx]  # (288,384) без парсинга
```
