Project Page: https://cal-cs180.github.io/fa24/hw/proj5/partb_fm.html

## Part 1: Training a Single-Step Denoising UNet
Unconditional UNet Architecture

![](./UNet%20Architecture/unconditional_arch.png)

1. A visualization of the noising process using $\sigma$=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
![alt text](./Deliverables/1-1.png)
2. A training loss curve plot every few iterations during the whole training process
![alt text](./Deliverables/1-2.png)
3. Sample results on the test set after the first and the 5-th epoch
    * First epoch
    
        ![alt text](./Deliverables/1-3.png)
    * Fifth epoch

        ![alt text](./Deliverables/1-3-2.png)
4. Sample results on the test set with out-of-distribution noise levels after the model is trained. Keep the same image and vary $\sigma$=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
![alt text](./Deliverables/1-4.png)
5. Sample results on the test set with pure noise $\epsilon\sim{N(0,I)}$
    * First epoch
        
        ![alt text](./Deliverables/1-5-1.png)
    * Fifth epoch 
    
        ![alt text](./Deliverables/1-5-2.png)
6. Average image of the training set along with a brief description comparing it to the denoising results.
![alt text](./Deliverables/1-6.png)

Description: The denoised result is still pure noise, the model is unable to denoise pure noise into anything meaningful. While comparing them with the average image, we could kinda see a very vague pattern on the denoised output similar to the average image.

## Part 2: Training a Time-Conditioned Denoising UNet
Time conditioned UNet Architecture

![](./UNet%20Architecture/conditional_arch_fm.png)

1. A training loss curve plot for the time-conditioned UNet over the whole training process
![](./Deliverables/2-1.png)

2. Sampling results for the time-conditioned UNet for 5 and 10 epochs.
![](./Deliverables/2-2.png)

## Part 2: Training a Time&Class -Conditioned Denoising UNet
Time&Class conditioned UNet Architecture

![](./UNet%20Architecture/time_added_arch.png)

1. A training loss curve plot for the class-conditioned UNet over the whole training process(10 epochs)
![](./Deliverables/2-3-training-epoch10.png)

2. Sampling results for the class-conditioned UNet for 5 and 10 epochs.
![](./Deliverables/2-4-epoch10-2.png)