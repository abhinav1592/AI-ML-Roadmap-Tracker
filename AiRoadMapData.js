// AiRoadMapData.js
const roadmapData = [
  {
    id: "phase1",
    title: "Phase 1: Mathematical Foundations & Data Science Fundamentals",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase1_week1",
        title: "Week 1: Statistics & Probability Fundamentals",
        dailyTasks: [
          { id: "p1w1t1_day1-2", description: "Khan Academy Statistics - Descriptive statistics, distributions", resources: [{ name: "Khan Academy Statistics", url: "https://www.khanacademy.org/math/statistics-probability" }] },
          { id: "p1w1t1_day3-4", description: "Think Stats 2e - Chapters 1-3 (Exploratory Data Analysis)", resources: [{ name: "Think Stats 2e", url: "https://greenteapress.com/wp/think-stats-2e/" }] },
          { id: "p1w1t1_day5-7", description: "Hands-on with NumPy and SciPy stats", resources: [{ name: "NumPy Docs", url: "https://numpy.org/doc/" }, { name: "SciPy Stats Docs", url: "https://docs.scipy.org/doc/scipy/reference/stats.html" }] },
          { id: "p1w1t1_project", description: "Project: Create a statistical analysis dashboard using Streamlit", resources: [{ name: "Streamlit Docs", url: "https://streamlit.io/" }, { name: "UCI Adult Income Dataset", url: "https://archive.ics.uci.edu/ml/datasets/Adult" }] }
        ]
      },
      {
        id: "phase1_week2",
        title: "Week 2: Linear Algebra for ML",
        dailyTasks: [
          { id: "p1w2t1_day1-3", description: "3Blue1Brown Linear Algebra Series - Watch 2 videos daily", resources: [{ name: "3Blue1Brown Linear Algebra", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi" }] },
          { id: "p1w2t1_day4-5", description: "MIT Linear Algebra Notes - Focus on eigenvectors, matrix decomposition", resources: [{ name: "MIT Linear Algebra", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/" }] },
          { id: "p1w2t1_day6-7", description: "Implement from scratch using NumPy", resources: [{ name: "NumPy Docs", url: "https://numpy.org/doc/" }] },
          { id: "p1w2t1_project", description: "Project: Build a simple recommendation system using matrix factorization", resources: [{ name: "MovieLens 100K Dataset", url: "https://grouplens.org/datasets/movielens/100k/" }] }
        ]
      },
      {
        id: "phase1_week3",
        title: "Week 3: Calculus & Optimization",
        dailyTasks: [
          { id: "p1w3t1_day1-3", description: "Paul's Online Math Notes - Calculus - Derivatives, partial derivatives", resources: [{ name: "Paul's Online Math Notes", url: "https://tutorial.math.lamar.edu/Classes/CalcI/CalcI.aspx" }] },
          { id: "p1w3t1_day4-5", description: "Gradient descent intuition via Distill.pub articles", resources: [{ name: "Distill.pub Gradient Descent", url: "https://distill.pub/2017/momentum/" }] },
          { id: "p1w3t1_day6-7", description: "Implement optimization algorithms", resources: [] },
          { id: "p1w3t1_project", description: "Project: Visualize gradient descent convergence on different loss functions", resources: [{ name: "Matplotlib Docs", url: "https://matplotlib.org/" }] }
        ]
      },
      {
        id: "phase1_week4",
        title: "Week 4: Data Manipulation & EDA Mastery",
        dailyTasks: [
          { id: "p1w4t1_day1-2", description: "Advanced Pandas techniques - MultiIndex, GroupBy operations", resources: [{ name: "Pandas Docs", url: "https://pandas.pydata.org/docs/" }] },
          { id: "p1w4t1_day3-4", description: "Seaborn and Plotly for advanced visualizations", resources: [{ name: "Seaborn Docs", url: "https://seaborn.pydata.org/" }, { name: "Plotly Docs", url: "https://plotly.com/python/" }] },
          { id: "p1w4t1_day5-7", description: "Complete EDA project", resources: [] },
          { id: "p1w4t1_project", description: "Comprehensive analysis of e-commerce data with insights", resources: [{ name: "E-commerce Dataset - Kaggle", url: "https://www.kaggle.com/datasets/carrieok/ecommerce-data" }] }
        ]
      }
    ]
  },
  {
    id: "phase2",
    title: "Phase 2: Machine Learning Foundations",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase2_week5",
        title: "Week 5: Supervised Learning Fundamentals",
        dailyTasks: [
          { id: "p2w5t1_day1-2", description: "Scikit-learn Course - Linear models, decision trees", resources: [{ name: "Scikit-learn Docs", url: "https://scikit-learn.org/stable/" }] },
          { id: "p2w5t1_day3-4", description: "Andrew Ng's ML Course - Week 2-3 - Theory + math", resources: [{ name: "Andrew Ng's ML Course", url: "https://www.coursera.org/learn/machine-learning" }] },
          { id: "p2w5t1_day5-7", description: "Hands-on implementation", resources: [] },
          { id: "p2w5t1_project", description: "Predict house prices with feature engineering", resources: [{ name: "Boston Housing Dataset", url: "https://www.kaggle.com/datasets/altavish/boston-housing-dataset" }] }
        ]
      },
      {
        id: "phase2_week6",
        title: "Week 6: Advanced Supervised Learning",
        dailyTasks: [
          { id: "p2w6t1_day1-2", description: "Ensemble methods - Random Forest, Gradient Boosting", resources: [{ name: "Ensemble Methods Guide", url: "https://scikit-learn.org/stable/modules/ensemble.html" }] },
          { id: "p2w6t1_day3-4", description: "XGBoost and LightGBM tutorials", resources: [{ name: "XGBoost Docs", url: "https://xgboost.readthedocs.io/en/stable/" }, { name: "LightGBM Docs", url: "https://lightgbm.readthedocs.io/en/latest/" }] },
          { id: "p2w6t1_day5-7", description: "Classification project", resources: [] },
          { id: "p2w6t1_project", description: "Customer churn prediction with model comparison", resources: [{ name: "Telco Customer Churn", url: "https://www.kaggle.com/datasets/blastchar/telco-customer-churn" }] }
        ]
      },
      {
        id: "phase2_week7",
        title: "Week 7: Unsupervised Learning & Clustering",
        dailyTasks: [
          { id: "p2w7t1_day1-2", description: "K-means, hierarchical clustering theory and implementation", resources: [{ name: "Scikit-learn Clustering", url: "https://scikit-learn.org/stable/modules/clustering.html" }] },
          { id: "p2w7t1_day3-4", description: "PCA and t-SNE for dimensionality reduction", resources: [{ name: "Scikit-learn PCA", url: "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html" }] },
          { id: "p2w7t1_day5-7", description: "Clustering project", resources: [] },
          { id: "p2w7t1_project", description: "Customer segmentation analysis", resources: [{ name: "Online Retail Dataset", url: "https://archive.ics.uci.edu/ml/datasets/Online+Retail" }] }
        ]
      },
      {
        id: "phase2_week8",
        title: "Week 8: Model Evaluation & MLOps Basics",
        dailyTasks: [
          { id: "p2w8t1_day1-2", description: "Cross-validation, hyperparameter tuning with Optuna", resources: [{ name: "Optuna Docs", url: "https://optuna.org/" }] },
          { id: "p2w8t1_day3-4", description: "Model versioning with MLflow basics", resources: [{ name: "MLflow Docs", url: "https://mlflow.org/docs/latest/index.html" }] },
          { id: "p2w8t1_day5-7", description: "End-to-end ML pipeline", resources: [] },
          { id: "p2w8t1_project", description: "Project: Automated model training and evaluation pipeline", resources: [{ name: "GitHub Actions", url: "https://docs.github.com/en/actions" }] }
        ]
      }
    ]
  },
  {
    id: "phase3",
    title: "Phase 3: Deep Learning Fundamentals",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase3_week9",
        title: "Week 9: Neural Networks from Scratch",
        dailyTasks: [
          { id: "p3w9t1_day1-3", description: "Neural Networks and Deep Learning Course - Week 1-2", resources: [{ name: "Neural Networks and Deep Learning", url: "http://neuralnetworksanddeeplearning.com/" }] },
          { id: "p3w9t1_day4-5", description: "Implement neural network from scratch using NumPy", resources: [{ name: "NumPy Docs", url: "https://numpy.org/doc/" }] },
          { id: "p3w9t1_day6-7", description: "Theory deep dive", resources: [] },
          { id: "p3w9t1_project", description: "Project: Handwritten digit recognition from scratch", resources: [{ name: "MNIST Dataset", url: "http://yann.lecun.com/exdb/mnist/" }] }
        ]
      },
      {
        id: "phase3_week10",
        title: "Week 10: TensorFlow/Keras Mastery",
        dailyTasks: [
          { id: "p3w10t1_day1-2", description: "TensorFlow Developer Certificate Path - Intro modules", resources: [{ name: "TensorFlow Tutorials", url: "https://www.tensorflow.org/tutorials" }] },
          { id: "p3w10t1_day3-4", description: "CNN fundamentals and implementation", resources: [{ name: "Keras Documentation", url: "https://keras.io/" }] },
          { id: "p3w10t1_day5-7", description: "Computer vision project", resources: [] },
          { id: "p3w10t1_project", description: "Image classification web app", resources: [{ name: "CIFAR-10 Dataset", url: "https://www.cs.toronto.edu/~kriz/cifar.html" }] }
        ]
      },
      {
        id: "phase3_week11",
        title: "Week 11: Advanced Deep Learning Architectures",
        dailyTasks: [
          { id: "p3w11t1_day1-2", description: "RNN, LSTM theory and implementation", resources: [{ name: "Understanding LSTMs", url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/" }] },
          { id: "p3w11t1_day3-4", description: "Transfer learning with pre-trained models", resources: [{ name: "Transfer Learning Guide", url: "https://www.tensorflow.org/tutorials/images/transfer_learning" }] },
          { id: "p3w11t1_day5-7", description: "Sequence modeling project", resources: [] },
          { id: "p3w11t1_project", description: "Project: Stock price prediction or text generation", resources: [{ name: "Yahoo Finance API", url: "https://finance.yahoo.com/lookup" }] }
        ]
      },
      {
        id: "phase3_week12",
        title: "Week 12: PyTorch & Advanced Concepts",
        dailyTasks: [
          { id: "p3w12t1_day1-2", description: "PyTorch Tutorials - Basics and autograd", resources: [{ name: "PyTorch Tutorials", url: "https://pytorch.org/tutorials/" }] },
          { id: "p3w12t1_day3-4", description: "Compare TensorFlow vs PyTorch implementation", resources: [] },
          { id: "p3w12t1_day5-7", description: "Advanced project", resources: [] },
          { id: "p3w12t1_project", description: "Generative model (VAE or simple GAN)", resources: [{ name: "CelebA Dataset (subset)", url: "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" }] }
        ]
      }
    ]
  },
  {
    id: "phase4",
    title: "Phase 4: Natural Language Processing",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase4_week13",
        title: "Week 13: Traditional NLP Foundations",
        dailyTasks: [
          { id: "p4w13t1_day1-2", description: "Text preprocessing with NLTK and spacy.", resources: [{ name: "NLTK Book", url: "https://www.nltk.org/book/" }, { name: "spaCy Course", url: "https://course.spacy.io/" }] },
          { id: "p4w13t1_day3-4", description: "TF-IDF, word embeddings (Word2Vec)", resources: [] },
          { id: "p4w13t1_day5-7", description: "Text classification project", resources: [] },
          { id: "p4w13t1_project", description: "Sentiment analysis API", resources: [{ name: "IMDB Reviews Dataset", url: "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews" }] }
        ]
      },
      {
        id: "phase4_week14",
        title: "Week 14: Advanced NLP & Word Embeddings",
        dailyTasks: [
          { id: "p4w14t1_day1-2", description: "Glove and FastText embeddings", resources: [{ name: "GloVe", url: "https://nlp.stanford.edu/projects/glove/" }, { name: "FastText", url: "https://fasttext.cc/" }] },
          { id: "p4w14t1_day3-4", description: "Named Entity Recognition and dependency parsing", resources: [{ name: "NER with spaCy", url: "https://spacy.io/usage/linguistic-features#named-entities" }] },
          { id: "p4w14t1_day5-7", description: "Information extraction project", resources: [] },
          { id: "p4w14t1_project", description: "Resume parsing and skill extraction system", resources: [] }
        ]
      },
      {
        id: "phase4_week15",
        title: "Week 15: Transformer Architecture & BERT",
        dailyTasks: [
          { id: "p4w15t1_day1-2", description: "Attention Is All You Need Paper study + Illustrated Transformer", resources: [{ name: "Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762" }, { name: "Illustrated Transformer", url: "https://jalammar.github.io/illustrated-transformer/" }] },
          { id: "p4w15t1_day3-4", description: "Hugging Face Transformers library tutorial", resources: [{ name: "Hugging Face Course", url: "https://huggingface.co/course/chapter1/1" }] },
          { id: "p4w15t1_day5-7", description: "BERT fine-tuning project", resources: [] },
          { id: "p4w15t1_project", description: "Question-answering system", resources: [{ name: "SQuAD Dataset", url: "https://rajpurkar.github.io/SQuAD-explorer/" }] }
        ]
      },
      {
        id: "phase4_week16",
        title: "Week 16: Large Language Models & Applications",
        dailyTasks: [
          { id: "p4w16t1_day1-2", description: "GPT architecture understanding and OpenAI API", resources: [{ name: "OpenAI API", url: "https://openai.com/docs/api-reference" }] },
          { id: "p4w16t1_day3-4", description: "Prompt engineering and few-shot learning", resources: [] },
          { id: "p4w16t1_day5-7", description: "LLM application project", resources: [] },
          { id: "p4w16t1_project", description: "AI-powered content generation tool (blog writer, code generator, or chatbot)", resources: [] }
        ]
      }
    ]
  },
  {
    id: "phase5",
    title: "Phase 5: Computer Vision & Advanced AI",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase5_week17",
        title: "Week 17: Advanced Computer Vision",
        dailyTasks: [
          { id: "p5w17t1_day1-2", description: "OpenCV for image processing and feature detection", resources: [{ name: "OpenCV Tutorials", url: "https://docs.opencv.org/4.x/d9/df8/tutorial_root.html" }] },
          { id: "p5w17t1_day3-4", description: "Advanced CNN architectures (ResNet, EfficientNet, Vision Transformer)", resources: [{ name: "CS231n Computer Vision", url: "http://cs231n.stanford.edu/" }] },
          { id: "p5w17t1_day5-7", description: "Object detection project", resources: [] },
          { id: "p5w17t1_project", description: "Real-time object detection system", resources: [{ name: "YOLO", url: "https://pjreddie.com/darknet/yolo/" }, { name: "Detectron2", url: "https://github.com/facebookresearch/detectron2" }] }
        ]
      },
      {
        id: "phase5_week18",
        title: "Week 18: Generative AI & GANs",
        dailyTasks: [
          { id: "p5w18t1_day1-2", description: "GAN theory and original paper study", resources: [{ name: "GAN Specialization Coursera", url: "https://www.coursera.org/specializations/generative-adversarial-networks-gans" }] },
          { id: "p5w18t1_day3-4", description: "Implement DCGAN with PyTorch", resources: [] },
          { id: "p5w18t1_day5-7", description: "Creative AI project", resources: [] },
          { id: "p5w18t1_project", description: "Art generation or style transfer application", resources: [{ name: "StyleGAN", url: "https://github.com/NVlabs/stylegan" }] }
        ]
      },
      {
        id: "phase5_week19",
        title: "Week 19: AI in Production & Edge Computing",
        dailyTasks: [
          { id: "p5w19t1_day1-2", description: "Model optimization with TensorFlow Lite and ONNX", resources: [{ name: "TensorFlow Lite Guide", url: "https://www.tensorflow.org/lite/guide" }] },
          { id: "p5w19t1_day3-4", description: "Docker containerization for ML models", resources: [{ name: "Docker Docs", url: "https://docs.docker.com/" }] },
          { id: "p5w19t1_day5-7", description: "Edge deployment project", resources: [] },
          { id: "p5w19t1_project", description: "Mobile app with on-device AI inference", resources: [{ name: "TensorFlow Lite Mobile", url: "https://www.tensorflow.org/lite/guide/device" }] }
        ]
      },
      {
        id: "phase5_week20",
        title: "Week 20: Reinforcement Learning Basics",
        dailyTasks: [
          { id: "p5w20t1_day1-2", description: "RL fundamentals and OpenAI Gym", resources: [{ name: "OpenAI Gym", url: "https://gymnasium.farama.org/" }] },
          { id: "p5w20t1_day3-4", description: "Q-learning and Stable Baselines3", resources: [{ name: "Stable Baselines3", url: "https://stable-baselines3.readthedocs.io/en/master/" }] },
          { id: "p5w20t1_day5-7", description: "RL game project", resources: [] },
          { id: "p5w20t1_project", description: "Train agent to play classic game (CartPole, Breakout)", resources: [] }
        ]
      }
    ]
  },
  {
    id: "phase6",
    title: "Phase 6: MLOps & Production Systems",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase6_week21",
        title: "Week 21: Model Versioning & Experiment Tracking",
        dailyTasks: [
          { id: "p6w21t1_day1-2", description: "MLflow for experiment tracking and model registry", resources: [{ name: "MLflow Docs", url: "https://mlflow.org/docs/latest/index.html" }] },
          { id: "p6w21t1_day3-4", description: "Weights & Biases for advanced experiment management", resources: [{ name: "Weights & Biases", url: "https://wandb.ai/" }] },
          { id: "p6w21t1_day5-7", description: "ML pipeline project", resources: [] },
          { id: "p6w21t1_project", description: "Automated model training pipeline with A/B testing", resources: [{ name: "Apache Airflow", url: "https://airflow.apache.org/" }, { name: "Prefect", url: "https://www.prefect.io/" }] }
        ]
      },
      {
        id: "phase6_week22",
        title: "Week 22: Model Deployment & Serving",
        dailyTasks: [
          { id: "p6w22t1_day1-2", description: "Docker and Kubernetes basics for ML", resources: [{ name: "Docker Docs", url: "https://docs.docker.com/" }, { name: "Kubernetes Docs", url: "https://kubernetes.io/docs/home/" }] },
          { id: "p6w22t1_day3-4", description: "Model serving with TensorFlow Serving or Seldon", resources: [{ name: "TensorFlow Serving", url: "https://www.tensorflow.org/tfx/guide/serving" }, { name: "Seldon Core", url: "https://www.seldon.io/tech/products/core/" }] },
          { id: "p6w22t1_day5-7", description: "Production deployment project", resources: [] },
          { id: "p6w22t1_project", description: "Scalable ML API with load balancing", resources: [{ name: "AWS", url: "https://aws.amazon.com/machine-learning/" }, { name: "Google Cloud AI Platform", url: "https://cloud.google.com/ai-platform" }] }
        ]
      },
      {
        id: "phase6_week23",
        title: "Week 23: Monitoring & Model Governance",
        dailyTasks: [
          { id: "p6w23t1_day1-2", description: "Model drift detection and data quality monitoring", resources: [] },
          { id: "p6w23t1_day3-4", description: "Evidently AI for ML monitoring", resources: [{ name: "Evidently AI", url: "https://evidentlyai.com/" }] },
          { id: "p6w23t1_day5-7", description: "Complete MLOps project", resources: [] },
          { id: "p6w23t1_project", description: "End-to-end ML system with monitoring dashboard", resources: [{ name: "Grafana", url: "https://grafana.com/" }] }
        ]
      },
      {
        id: "phase6_week24",
        title: "Week 24: CI/CD for Machine Learning",
        dailyTasks: [
          { id: "p6w24t1_day1-2", description: "GitHub Actions for ML workflows", resources: [{ name: "GitHub Actions for ML", url: "https://github.com/features/actions" }] },
          { id: "p6w24t1_day3-4", description: "Testing ML models and data validation", resources: [] },
          { id: "p6w24t1_day5-7", description: "Complete CI/CD pipeline", resources: [] },
          { id: "p6w24t1_project", description: "Automated ML pipeline from training to deployment", resources: [{ name: "DVC for Data Versioning", url: "https://dvc.org/" }] }
        ]
      }
    ]
  },
  {
    id: "phase7",
    title: "Phase 7: Specialized AI Applications",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase7_week25",
        title: "Week 25: Recommendation Systems",
        dailyTasks: [
          { id: "p7w25t1_day1-2", description: "Collaborative filtering and content-based approaches", resources: [{ name: "Recommender Systems Handbook", url: "https://dl.acm.org/doi/book/10.1007/978-0-387-85820-3" }] },
          { id: "p7w25t1_day3-4", description: "Deep learning for recommendations (Neural Collaborative Filtering)", resources: [] },
          { id: "p7w25t1_day5-7", description: "E-commerce recommendation project", resources: [] },
          { id: "p7w25t1_project", description: "Multi-algorithm recommendation engine", resources: [{ name: "Amazon Product Data", url: "https://nijian.github.io/amazon/" }] }
        ]
      },
      {
        id: "phase7_week26",
        title: "Week 26: Time Series & Forecasting",
        dailyTasks: [
          { id: "p7w26t1_day1-2", description: "ARIMA, seasonal decomposition with statsmodels", resources: [{ name: "statsmodels Docs", url: "https://www.statsmodels.org/stable/index.html" }] },
          { id: "p7w26t1_day3-4", description: "Prophet and deep learning approaches (LSTM, Transformer)", resources: [{ name: "Prophet by Facebook", url: "https://facebook.github.io/prophet/" }] },
          { id: "p7w26t1_day5-7", description: "Business forecasting project", resources: [] },
          { id: "p7w26t1_project", description: "Multi-variate time series forecasting dashboard", resources: [{ name: "Streamlit Docs", url: "https://streamlit.io/" }] }
        ]
      },
      {
        id: "phase7_week27",
        title: "Week 27: AI Ethics & Explainable AI",
        dailyTasks: [
          { id: "p7w27t1_day1-2", description: "Bias detection and fairness metrics in ML models", resources: [{ name: "AI Ethics Course - MIT", url: "https://ocw.mit.edu/courses/esd-74-ai-ethics-and-governance-fall-2020/" }] },
          { id: "p7w27t1_day3-4", description: "SHAP and LIME for model interpretability", resources: [{ name: "SHAP", url: "https://shap.readthedocs.io/en/latest/" }, { name: "LIME", url: "https://github.com/marcotcr/lime" }] },
          { id: "p7w27t1_day5-7", description: "Responsible AI project", resources: [] },
          { id: "p7w27t1_project", description: "Bias-aware hiring recommendation system", resources: [{ name: "Fairlearn", url: "https://fairlearn.org/" }, { name: "AI Fairness 360", url: "https://aif360.mybluemix.net/" }] }
        ]
      },
      {
        id: "phase7_week28",
        title: "Week 28: Advanced AI Research & Implementation",
        dailyTasks: [
          { id: "p7w28t1_day1-2", description: "Read and implement recent AI papers from Papers With Code", resources: [{ name: "Papers With Code", url: "https://paperswithcode.com/" }] },
          { id: "p7w28t1_day3-4", description: "Contribute to open-source AI projects", resources: [] },
          { id: "p7w28t1_day5-7", description: "Research implementation project", resources: [] },
          { id: "p7w28t1_project", description: "Implement and improve upon a recent AI research paper", resources: [] }
        ]
      }
    ]
  },
  {
    id: "phase8",
    title: "Phase 8: AI Leadership & Advanced Applications",
    durationWeeks: 4,
    weeks: [
      {
        id: "phase8_week29",
        title: "Week 29: Multi-modal AI & Advanced Architectures",
        dailyTasks: [
          { id: "p8w29t1_day1-2", description: "Vision-Language models (CLIP, DALLE)", resources: [{ name: "CLIP", url: "https://openai.com/research/clip" }, { name: "DALL-E", url: "https://openai.com/dall-e-2" }] },
          { id: "p8w29t1_day3-4", description: "Diffusion Models and latest generative AI trends", resources: [{ name: "Diffusion Models Tutorial", url: "https://jalammar.github.io/illustrated-diffusion/" }] },
          { id: "p8w29t1_day5-7", description: "Cutting-edge AI project", resources: [] },
          { id: "p8w29t1_project", description: "Multi-modal application (text-to-image, image captioning, or video analysis)", resources: [{ name: "Hugging Face Models", url: "https://huggingface.co/models" }] }
        ]
      },
      {
        id: "phase8_week30",
        title: "Week 30: AI Strategy & Business Applications",
        dailyTasks: [
          { id: "p8w30t1_day1-2", description: "AI strategy frameworks and ROI calculation", resources: [{ name: "AI Strategy Course - MIT Sloan", url: "https://mitsloan.mit.edu/executive-education/ai-strategy-and-business-applications" }] },
          { id: "p8w30t1_day3-4", description: "Case studies of successful AI implementations", resources: [{ name: "Harvard Business Review AI Articles", url: "https://hbr.org/topic/artificial-intelligence" }] },
          { id: "p8w30t1_day5-7", description: "Business AI solution", resources: [] },
          { id: "p8w30t1_project", description: "Complete AI solution for a business problem", resources: [] }
        ]
      },
      {
        id: "phase8_week31",
        title: "Week 31: Team Leadership & Mentoring",
        dailyTasks: [
          { id: "p8w31t1_day1-2", description: "Technical leadership in AI teams", resources: [] },
          { id: "p8w31t1_day3-4", description: "Create learning materials and tutorials", resources: [] },
          { id: "p8w31t1_day5-7", description: "Community contribution", resources: [] },
          { id: "p8w31t1_project", description: "Mentor junior developers or contribute to AI education", resources: [{ name: "Kaggle Learn", url: "https://www.kaggle.com/learn" }] }
            ]
      },
      {
        id: "phase8_week32",
        title: "Week 32: Portfolio Optimization & Career Acceleration",
        dailyTasks: [
          { id: "p8w32t1_day1-2", description: "Optimize GitHub portfolio and LinkedIn profile", resources: [{ name: "GitHub", url: "https://github.com/" }, { name: "LinkedIn", url: "https://www.linkedin.com/" }] },
          { id: "p8w32t1_day3-4", description: "Create impressive demo videos and documentation", resources: [] },
          { id: "p8w32t1_day5-7", description: "Career planning and networking", resources: [] },
          { id: "p8w32t1_activities", description: "Activities: Apply advanced AI skills to current role", resources: [{ name: "NeurIPS", url: "https://neurips.cc/" }, { name: "ICML", url: "https://icml.cc/" }] }
        ]
      }
    ]
  }
];

export default roadmapData;

