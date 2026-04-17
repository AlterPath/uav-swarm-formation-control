# swarm_pkg — Рой квадрокоптеров с APF

Программно-алгоритмическое обеспечение системы управления группой
квадрокоптеров (PX4 + Gazebo, модель iris) для аэрофотосъёмки.

---

## Структура пакета

```
swarm_pkg/
├── src/
│   ├── drone_node.py        ← главный ROS-узел (один экземпляр на дрон)
│   ├── apf.py               ← метод искусственных потенциальных полей
│   ├── mission_planner.py   ← планировщик маршрута (змейка)
│   ├── formation_manager.py ← мониторинг строя и отказоустойчивость
│   └── logger.py            ← CSV-лог снимков и пропусков
├── launch/
│   └── swarm.launch         ← запуск роя (3 дрона по умолчанию)
├── CMakeLists.txt
└── package.xml
```

---

## Архитектура APF

Результирующий вектор скорости каждого дрона:

```
v = F_att(pos, target)                 ← аттракция к цели
  + Σ F_rep_drone(pos, nb_pos)         ← репульсия от соседей
  + Σ (F_rep_obs + F_tang)(pos, obs)   ← репульсия + тангенция от препятствий
```

**Тангенциальная составляющая** у препятствий:
- чётный drone_id → обход слева  (+sign)
- нечётный drone_id → обход справа (−sign)

Это предотвращает резонансный цикл, когда два дрона "застревают"
по разные стороны препятствия и непрерывно переключаются.

---

## Конечный автомат дрона

```
INIT ──▶ ARMING ──▶ TAKEOFF ──▶ SYNC ──▶ MISSION ──▶ LAND
                                  │
                          барьер готовности
                    (все дроны публикуют /swarm/ready/uav{id})
```

**MISSION** использует два режима навигации:
- `navigate_survey` — точки съёмки (таймаут 15 с, пропуски логируются)
- `navigate_transit` — перелёты между рядами (таймаут 30 с, пропуски не логируются)

---

## Отказоустойчивость

Каждый дрон публикует heartbeat в `/swarm/heartbeat/uav{id}` с частотой 2 Гц.
При потере heartbeat > 3 с дрон объявляется выбывшим.

Перестройка строя:
```
h_new = h_base × √(n_total / n_active)
w_new = w_base × √(n_total / n_active)
```
Суммарное покрытие сохраняется.

---

## Топики

| Топик | Тип | Направление |
|-------|-----|-------------|
| `/uav{id}/mavros/state` | `mavros_msgs/State` | подписка |
| `/uav{id}/mavros/local_position/pose` | `PoseStamped` | подписка |
| `/uav{id}/mavros/setpoint_raw/local` | `PositionTarget` | публикация |
| `/swarm/position/uav{id}` | `PoseStamped` | публикация + подписка |
| `/swarm/heartbeat/uav{id}` | `Header` | публикация + подписка |
| `/swarm/ready/uav{id}` | `Header` | публикация + подписка |
| `/iris{id}/usb_cam/image_raw` | `sensor_msgs/Image` | подписка |

---

## Быстрый старт

### 1. Сборка

```bash
cd ~/catkin_ws/src
ln -s /path/to/swarm_pkg .
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 2. Запуск симуляции (PX4 + Gazebo)

```bash
# 3 дрона в пустой среде
cd ~/PX4-Autopilot
DONT_RUN=1 make px4_sitl_default gazebo
source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default

# Запустить 3 iris с MAVROS
roslaunch px4 multi_uav_mavros_sitl.launch
```

### 3. Запуск роя

```bash
roslaunch swarm_pkg swarm.launch \
  n:=3 \
  altitude:=10.0 \
  area_xmin:=0.0 area_xmax:=60.0 \
  area_ymin:=0.0 area_ymax:=60.0 \
  fov_deg:=90.0 \
  overlap:=0.15 \
  log_dir:=/tmp/swarm_logs
```

### 4. Добавление препятствий

В launch-файле или через параметрсервер передайте координаты препятствий
в виде списка [[x,y,z], ...]:

```bash
rosparam set /drone_node_0/obstacles "[[10,10,0],[25,30,0]]"
```

---

## Логи

После посадки в `log_dir` появятся файлы:

```
/tmp/swarm_logs/
├── uav0_photos_20250418_120000.csv   ← сделанные снимки
├── uav0_skipped_20250418_120000.csv  ← пропущенные точки
├── uav1_photos_...csv
└── uav0_photos/                      ← JPEG-кадры с камеры
    ├── uav0_20250418_120015_0000.jpg
    └── ...
```

Формат `*_photos.csv`:
```
timestamp, filename, target_x, target_y, target_z, actual_x, actual_y, actual_z, distance_error
```

Формат `*_skipped.csv`:
```
timestamp, reason, target_x, target_y, target_z, drone_x, drone_y, drone_z
```
