# Background and Project Overview

Much of daily life assumes that people can freely use both arms, locate objects in front of them, and reach naturally for what they need. For people with visual or upper-limb impairments, this assumption does not hold. This project develops a wearable VLA (Vision-Language-Action) assistant based on Gemma 4 that serves as a “second arm” for these users, together with the data collection, generation, and annotation infrastructure needed to train it as an integrated system.

# Why a wearable form factor

Daily-life assistance often consists of small physical actions: pulling out a chair, opening a door, taking an item from a shelf, picking up a dropped object, retrieving something from a bag, or holding a container steady. These actions are not a fixed task set; they arise unpredictably across everyday situations. Rather than a large robot that automates housework, users need a system that moves with them and assists with the specific action needed at that moment. This motivates a wearable form factor that stays close to the user’s body.

# The challenges this introduces

Wearability increases flexibility but also raises the technical difficulty. Because target actions arise across daily life, hand-engineering a small set of predefined tasks is insufficient. The system must operate safely near the body, handle a constantly shifting first-person camera viewpoint, work within the arm’s limited reachable range, and remain lightweight enough for local execution without relying on a large cloud-hosted model.

# Approach: model and data as two pillars

The project addresses these challenges through both model design and data infrastructure.

On the model side, we built a VLA model that uses Gemma 4’s visual and language understanding to generate robot arm actions from the user’s words and surrounding context. With the lightweight Gemma 4-E2B backbone, the model combines the recognition and action generation capabilities required for daily-life assistance while keeping local execution feasible. Effective learning from diverse data was also a core design requirement.

On the data side, we focused on first-person human demonstration data. Direct teleoperation, collected episode by episode, cannot cover the countless support situations found in daily life. By capturing natural human actions in everyday environments and converting them into robot-training data, we can obtain a broader range of support tasks more naturally and at greater scale. To enable this, we developed a pipeline for converting human demonstrations into robot-training data, MimicAnno for subtask labeling and object annotation, and MimicRec for integrating data collection, management, and inference interfaces. Together, these components form a platform that connects data collection, annotation, training, evaluation, and re-training into a continuous development loop.

# The value of this project

The significance of this project is not simply that it moves a robot arm. It combines a Gemma 4-based wearable VLA architecture that learns from diverse data with a scalable, human-demonstration-driven data generation and training pipeline designed for the diversity of daily-life support tasks. Rather than a one-off demo, we implemented a VLA system that continuously learns from everyday motions and responds to previously unseen support situations.

# Project Overview

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2F6227624cde4a8fcec774ffb28add6a9f%2FGEM4.jpg?generation=1779067763846631&alt=media)

The user gives a voice instruction starting with “hey GEM.” Once this is detected, the audio file is converted into text, embedded into the prompt, and input to the VLA together with images and joint information. MimicRec functions both as an interface connecting the VLA and the robot and as a data collection application. MimicAnno is responsible for annotating these robot data and generating robot data from human demonstrations.

# VisionLanguageActionModel

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2F0ebdb094a641845b5f8c51aa28865610%2FVLA_archi.jpg?generation=1778807945930266&alt=media)

In this model, the action-generation adapter performs cross-attention over the hidden states of each layer of Gemma 4, enabling the model to generate actions while efficiently leveraging Gemma 4’s visual and language understanding. In addition, by switching the input and output projectors for each domain, the model is expected to acquire domain-independent shared knowledge and representations through pretraining on diverse robot datasets.

## Training
For pretraining, we used multiple robot datasets from Open-X-Embodiment and the LIBERO dataset at a ratio of 8:2. We then fine-tuned the model using real-world robot data and the four LIBERO suites.

## L1-Loss-Curve (Pretrain+FT)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2Fdff2f98b168ab2a928dc780f5fe270b1%2F2026-05-18%20170709.png?generation=1779091703941535&alt=media)

The loss curves also show that pretraining and subsequent fine-tuning based on the knowledge acquired during pretraining are functioning effectively.

## benchmark (LIBERO-4suites-SuccessRate 50k step)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2F55b8b5136effcefa428d7908eb79088c%2F2026-05-18%20183655.png?generation=1779097033684182&alt=media)

### LIBERO-Long-Task9
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2F221d5d989e3f54e4b8ed4602f67f0c71%2F20260514-1917-37.4526929%20(3).gif?generation=1778827426884787&alt=media)

The Object and Goal tasks achieved high success rates, indicating that the model can effectively extract useful knowledge from Gemma 4’s language and visual understanding capabilities. The procedures for reproducing each experiment are described in the README at https://github.com/takaki-maeda-99/GEM-4-VLA.

# MimicRec

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2F0abf26307947a733db6158d9561197aa%2F2026-05-15%20162001.png?generation=1778829628702224&alt=media)

MimicRec provides functions for data collection, visualization, upload, and inference interfaces. By abstracting the interface with the robot, the system is designed so that data collection and inference can be performed through a common framework across different robots, simply by adding a control adapter for each robot. Because the application is intended to be used for extended periods of time, we also placed strong emphasis on UX. We will continue improving its functionality going forward.

Demo site: https://takaki-maeda-99.github.io/MimicRec/

# MimicAnno

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20657757%2F27c7a5ab9bdddc0f138fdcf5a2e75c98%2FmimicannoUIsample.png?generation=1779025174828328&alt=media)

MimicAnno is a pipeline that automatically annotates subtask labels on robot demonstration data, paired with a second pipeline that estimates end-effector trajectories from first-person human videos.

### Subtask annotation. 

Subtask boundaries are detected from gripper open/close events, EEF velocity and acceleration, and changes in action norm — with gripper state as the strongest signal, since grasp/release timing tends to align with subtask transitions. Each segment is then tracked with SAM3 and passed to a Gemma 4 model, QLoRA-finetuned with Unsloth, which generates the subtask. For a tape-in-bottle task, the pipeline produced approach_object → grasp → move_to_target → place_object → retreat, each tagged with a verb, object, target, and confidence score.


### EEF estimation. 
A first-person video from a GoPro Hero 11 Black (Max Lens mode) is processed with MediaPipe for hand landmarks and UniDAC for metric depth. Wrist orientation is reconstructed from the landmark geometry, and depth supplies the metric scale, yielding per-frame wrist position (m), orientation (yaw/pitch/roll), and pinch distance (mm) in the camera frame.


The two pipelines are designed around a shared data format, so EEF estimates can flow directly into the annotation pipeline — opening a path to generating subtask-labeled robot data end-to-end from first-person human video alone.


# Safety, Scope, and Positioning of This Project

This project is a research prototype and is not a medical device or a certified accessibility assistive device. We do not make any claims regarding clinical effectiveness, and the system has not obtained regulatory approval as a medical device or certification as an accessibility assistive device.

This project is an open implementation for research purposes, intended to explore what forms of physical assistance may become possible in the future for users with visual or upper-limb impairments. At this stage, it is not positioned as a product intended for real-world deployment, but as a research prototype for validating technical feasibility.

# What Remains to Be Done and Future Directions

In this project, we integrated the core components of a wearable VLA system, including the data collection infrastructure, annotation tools, and real-robot interface. At the same time, there remain aspects that have not yet been fully validated, as well as areas that require further improvement.

First, although cross-embodiment learning is supported in the codebase, we have not yet conducted large-scale validation across multiple robot embodiments. Future work should more systematically evaluate the extent to which shared knowledge and representations can be transferred across different robots.

Second, the current VLA model still has room for improvement. In particular, its performance on Long tasks in the LIBERO benchmark remains limited, and the ability to reliably connect multiple subtasks is an important direction for future work.

Third, the wearable hardware is still at the prototype stage and has been designed around a single user. We have not yet conducted cross-subject evaluations of the wearing form or physical fit. Future evaluations should consider differences in body shape, ease of wearing and removal, and the physical burden of long-term use.

Moving forward, we plan to pursue large-scale pretraining using robot data generated from human demonstration data, with the goal of improving generalization to diverse assistive actions in everyday environments. We also plan to introduce hierarchical reasoning based on subtask inference for long-horizon tasks, and to extend the model to incorporate multimodal inputs such as object detection results.

On the hardware side, we will continue developing a wearable mechanism that is less dependent on a specific user and easier to put on and take off. In addition, we will improve the efficiency of local inference on Jetson through optimization techniques such as quantization, reducing both latency and memory usage and making cloud-independent execution more practical