# Ghost-LSNet: See Large Focus Small for WLANs 

## ✨ IMPORTANT NOTES
The **LSNet** and **GLSNet (Ghost-LSNet)** implemented in this repository are **adjusted, simplified, and preliminary experimental versions** designed for WLAN (Wireless Local Area Network) signal processing tasks.  
These implementations are NOT the complete/accurate replication of the original "See Large Focus Small" LSNet architecture, and are still pending further optimization and refinement (for research and testing purposes only).
They are NOT the complete/accurate replication of the original "See Large Focus Small" LSNet architecture. For the official, full, and precise implementation of LSNet, please refer to:
- Official GitHub Repository: [https://github.com/THU-MIG/lsnet]
- Original LSNet Paper:       [https://arxiv.org/abs/2503.23135]
  LSNet: See Large, Focus Small. CVPR 2025.
  Ao Wang, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding

## ✨  Brief Introduction
1. **Models**:
*Note: All models above are implemented as 1D (one-dimensional) variants changed for sequential signal processing tasks.*
   - DNCNN
   - LSNet
   - GLSNet
2. **Performance Analysis**:
   - Ready-to-use plotting scripts for quantitative comparison (SNR, MSE, etc.) of all methods
3. **Dataset Support**:
   - Partial public datasets for ChB/ChD scenarios (40dB noise level) aligned with WLAN signal characteristics

## ✨ Prerequisites
### Environment Requirements
- Python 3.8 or higher
- (Optional) CUDA 11.0+ for GPU acceleration (compatible with Windows/Linux)
- pip install -r requirements.txt
