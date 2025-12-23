# AS_car_following
This repository contains a unique dataset of real-world autonomous shuttle (AS) trajectories collected in mixed-traffic conditions. 


# Autonomous Shuttle Car-Following Dataset and Models

This repository contains data and model files developed as part of the research by **Renan Favero** and **Lily Elefteriadou (2025)** at the **University of Florida**.  
The study focuses on **autonomous shuttle (AS)** car-following behavior, including calibrated models and cleaned trajectory data.

For details about the data collection and methodology, please refer to the publication:  
📄 [Favero, R., & Elefteriadou, L. (2025). IEEE Transactions on Intelligent Transportation Systems](https://ieeexplore.ieee.org/document/10960526)

---

## 📂 Dataset Overview

The file **`dataset_models`** contains the cleaned dataset used for training and testing.  
Below are the main variables included:

| Variable | Description |
|-----------|--------------|
| `Time_[s]` | Time of the trajectory (s) |
| `Leader_pos_[ft]` | Position of the leader vehicle (ft) |
| `Leader_sp_[ft]` | Speed of the leader vehicle (ft/s) |
| `delta_s` | Spacing between leader and follower (ft) |
| `delta_v` | Speed difference between leader and follower (ft/s) |
| `Follower_sp_[ft]` | Autonomous shuttle speed (ft/s) |
| `Follower_acc` | Real AS acceleration (ft/s²) |
| `Follower_pos_[ft]` | Real AS position (ft) |
| `Follower_sp_[ft]_t-1` | AS speed at previous time step (ft/s) |
| `Follower_pos_[ft]_t-1` | AS position at previous time step (ft) |
| `Follower_acc_t-1` | AS acceleration at previous time step (ft/s²) |
| `delta_t` | Time step interval (s) |
| `trajectory_id` | Unique trajectory identifier |

---

## ⚙️ Model Outputs

Each time step includes predicted accelerations from multiple models:acc_LGBM, acc_CNN, acc_Transformer, acc_Feedforward_NN, acc_XGBoost,
acc_RF, acc_LSTM, acc_SVM, acc_LightGBM, real_acc_idm, acc_idm



---

## 📁 Repository Structure

- **`dataset_models/`** – Cleaned main dataset used in the study.  
- **`results_<modelname>/`** – Folders containing calibrated model results for each ML algorithm.  
- **`Test_<modelname>.py`** – Python scripts for model inference and validation.  

---

## 🔒 License and Copyright

This dataset and accompanying code are protected by copyright © 2025  
**Renan Favero and Lily Elefteriadou, University of Florida**.  
Use of this material is permitted for research and educational purposes with proper citation.

Please cite as:  
> TBD

---

## 📫 Contact

For questions or collaborations, please contact:  
**Renan Favero**
**Email:** renanfavero30@hotmail.com



