# Ghost-LSNet: See Large Focus Small for WLANs

## ✨ IMPORTANT NOTES
The **LSNet** and **GLSNet (Ghost-LSNet)** implemented in this repository are **adjusted and simplified versions** designed for WLAN (Wireless Local Area Network) signal processing tasks.  
They are NOT the complete/accurate replication of the original "See Large Focus Small" LSNet architecture. For the official, full, and precise implementation of LSNet, please refer to:
- Official GitHub Repository: [https://github.com/xxx/LSNet-Official](https://github.com/xxx/LSNet-Official) (替换为真实的LSNet官方仓库链接)
- Original LSNet Paper:  
  Author, Title, Conference/Journal, Year.  
  (示例：Li X, Zhang Y, et al. "See Large Focus Small: LSNet for High-Resolution Feature Learning" [C]// CVPR 202X. IEEE, 202X.)

## ✨ Core Features
1. **Multi-Method Implementation**:
   - DNCNN (Baseline for WLAN signal denoising)
   - LSNet (Adjusted & simplified version for WLAN scenarios)
   - GLSNet (Ghost module enhanced LSNet, adapted for WLAN)
   - Wiener Filter (Classical signal denoising baseline)
2. **Performance Analysis**:
   - Ready-to-use plotting scripts for quantitative comparison (SNR, MSE, runtime, etc.) of all methods
3. **Dataset Support**:
   - Partial public datasets for ChB/ChD scenarios (40dB noise level) aligned with WLAN signal characteristics

## ✨ Prerequisites
### Environment Requirements
- Python 3.8 or higher
- (Optional) CUDA 11.0+ for GPU acceleration (compatible with Windows/Linux)

### Installation
1. Clone this repository:
   ```bash
   git clone [你的仓库链接].git
   cd Ghost-LSNet-WLAN
