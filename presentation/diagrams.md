# 📊 Mermaid Diagrams for Product Recommendation System

Use these codes to generate professional diagrams for your report or presentation.

---

## 1. System Architecture (High Level)
**Description:** Shows the 4 main layers of the application.

```mermaid
graph TD
    subgraph Client_Layer
        User[End User]
        Admin[Administrator]
    end

    subgraph Application_Layer
        API[Inference API]
        Dashboard[Analytics Dashboard]
    end

    subgraph Logic_Layer
        Pipeline[ETL Pipeline]
        ModelEngine[Recommendation Engine]
        Hybrid[Hybrid Strategy Switch]
    end

    subgraph Data_Layer
        RawData[(Raw CSV/Txt)]
        CleanData[(Cleaned Data)]
        Artifacts[(Model Artifacts .pkl)]
    end

    User -->|Requests Recs| API
    Admin -->|Triggers Training| Pipeline
    
    API --> Hybrid
    Hybrid --> ModelEngine
    
    Pipeline -->|Reads| RawData
    Pipeline -->|Saves| CleanData
    Pipeline -->|Trains| ModelEngine
    ModelEngine -->|Saves| Artifacts
```

---

## 2. Data Processing Workflow (ETL)
**Description:** The detailed steps from Raw Data to Training.

```mermaid
flowchart LR
    A[Raw Data] -->|Load| B(DataLoader)
    B -->|Combine| C{DataCombiner}
    C -->|Generate IDs| D[Combined Data]
    D -->|Clean| E(DataCleaner)
    
    subgraph Cleaning Steps
        E1[Impute Missing Ratings]
        E2[Fix Price Logic]
        E3[Convert Types]
    end
    
    E --> E1 --> E2 --> E3 --> F[Clean DataFrame]
    F -->|Sample| G(Sampler 25k)
    G -->|Feed| H[Model Training]
```

---

## 3. Class Diagram (Code Structure)
**Description:** How your Python classes relate to each other.

```mermaid
classDiagram
    class DataLoader {
        +load_amazon_data()
        +load_reviews()
    }
    class DataCombiner {
        +generate_ids()
        +assign_reviews()
    }
    class DataCleaner {
        +fix_price_logic()
        +handle_missing()
    }
    class EDA {
        +plot_distributions()
        +correlation_matrix()
    }
    class ModelTrainer {
        +train_knn()
        +train_svd()
        +grid_search()
    }

    DataLoader --> DataCombiner
    DataCombiner --> DataCleaner
    DataCleaner --> EDA
    DataCleaner --> ModelTrainer
```

---

## 4. Sequence Diagram (User Request)
**Description:** What happens precisely when a user asks for a recommendation.

```mermaid
sequenceDiagram
    actor U as User
    participant API as Recommender System
    participant KNN as KNN Model
    participant TFIDF as Content Engine

    U->>API: View Product "iPhone 13"
    API->>KNN: Get Neighbors(iPhone 13)
    
    alt KNN has Data (Warm Start)
        KNN-->>API: Returns [Case, Charger, AirPods]
    else KNN has No Data (Cold Start)
        KNN-->>API: Returns Empty/Low Confidence
        API->>TFIDF: Find Similar Text ("iPhone 13")
        TFIDF-->>API: Returns [Samsung S21, Pixel 6]
    end
    
    API-->>U: Show Top 6 Recommendations
```

---

## 5. Model Selection Logic (The Tournament)
**Description:** How the system decides which model is best.

```mermaid
stateDiagram-v2
    [*] --> Train_Baseline
    [*] --> Train_KNNBasic
    [*] --> Train_SVD
    [*] --> Train_KNNWithMeans

    Train_Baseline --> Calculate_RMSE
    Train_KNNBasic --> Calculate_RMSE
    Train_SVD --> Calculate_RMSE
    Train_KNNWithMeans --> Calculate_RMSE

    Calculate_RMSE --> Compare_Scores
    
    state "Select Best Model" as Select
    
    Compare_Scores --> Select: Lowest Error Wins
    
    Select --> Save_Artifacts
    Save_Artifacts --> [*]
```
