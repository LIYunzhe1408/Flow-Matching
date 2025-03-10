## Part 1: Training a Single-Step Denoising UNet
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