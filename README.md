# Bosch Applied Computer Vision Assignment

## Project Overview

This repository contains the implementation and analysis for the Bosch Applied Computer Vision assignment. The project is structured into multiple tasks focusing on data analysis, model training, and evaluation using computer vision techniques. Each task is encapsulated in its own directory with detailed instructions and code.

The goal is to analyze Bosch's dataset, build predictive models, and evaluate their performance with the provided tools and scripts.

---

## Repository Structure

```
bosch_applied_cv_assignment/
│
├── data/                       # Dataset files (to be set up by user)
│
├── tasks/
│   ├── 1_data_analysis/        # Task 1: Data exploration and preprocessing
│   │   └── README.md
│   ├── 2_model_training/       # Task 2: Model training and checkpoint saving
│   │   └── README.md
│   └── 3_evaluation/           # Task 3: Model evaluation and analysis
│       └── README.md
│
├── checkpoints/                # Saved model checkpoints (generated after training)
│
├── analysis_results/           # Output from analysis scripts and evaluation metrics
│
├── Dockerfile                  # Dockerfile for containerized environment
├── requirements.txt            # Python dependencies required for the project
└── README.md                   # This master README file
```

---

## Dataset Setup

1. Download the Bosch dataset from the official source or the link provided by your instructor.
2. Place the dataset files inside the `data/` directory at the root of the repository.
3. `data/` should have `bdd100k_images_100k` and `bdd100k_labels_release`

---

## Docker Setup

To ensure consistent environment setup, you can use the provided Docker container.

### Build Docker Image Locally

```bash
docker build -t bosch_cv_assignment .
```

### Pull Pre-built Docker Image from Docker Hub

```bash
docker pull ajwahir/bosch_cv_assignment:latest
```

### Run Docker Container

```bash
docker run -dit \                        
  --name bosch-assignment-fin \
  --shm-size=8g \
  -p 8501:8501 \
  -p 5181:5181 \
  -v $(pwd):/workspace \
  bosch-cv-fin:1.0.0 \
  tail -f /dev/null
```

To enter the conatainer
```docker exec -it bosch-assignment-fin bash```

---


## Running the Tasks

Each task has its own directory with detailed instructions. Please refer to their README files for step-by-step guides.

- [Task 1 - Data Analysis](tasks/1_data_analysis/README.md)
- [Task 2 - Model Training](tasks/2_model/README.md)
- [Task 3 - Evaluation](tasks/3_eval_viz/README.md)

---

