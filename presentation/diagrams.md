# System Diagrams

The following diagrams illustrate the architecture, workflow, and use cases of the Product Recommendation System.

## 1. System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
    end

    subgraph "Application Layer"
        Flask[Flask Web Server]
        Auth[Auth Module]
        Routes[API Routes]
        RecEngine[Recommendation Engine]
    end

    subgraph "Data Layer"
        MySQL[(MySQL Database)]
        ModelStore[Model Storage]
    end

    subgraph "ML Pipeline"
        DataLoad[Data Loader]
        Preprocess[Preprocessing]
        Train[Model Training]
        Eval[Evaluation]
    end

    Browser -- "HTTP Requests" --> Flask
    Flask -- "Authentication" --> Auth
    Flask -- "Routing" --> Routes
    Routes -- "Query Data" --> MySQL
    Routes -- "Get Recommendations" --> RecEngine
    RecEngine -- "Load Model" --> ModelStore
    
    DataLoad -- "Raw Data" --> Preprocess
    Preprocess -- "Clean Data" --> Train
    Train -- "Trained Model" --> ModelStore
    Train -- "Metrics" --> Eval
```

## 2. Workflow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Database
    participant RecModel

    User->>Frontend: Access Home Page
    Frontend->>Backend: GET /
    Backend->>Database: Fetch Popular Products
    Database-->>Backend: Return Products
    Backend-->>Frontend: Render Home Page
    
    User->>Frontend: Login(username, password)
    Frontend->>Backend: POST /login
    Backend->>Database: Validate Credentials
    Database-->>Backend: User Valid
    Backend-->>Frontend: Redirect to Dashboard
    
    User->>Frontend: View Product(id)
    Frontend->>Backend: GET /product/<id>
    Backend->>Database: Get Product Details
    Database-->>Backend: Product Info
    Backend->>RecModel: Get Recommendations(id)
    RecModel-->>Backend: List of Similar Products
    Backend-->>Frontend: Render Product Page with Recs
```

## 3. Use Case Diagram

```mermaid
usecaseDiagram
    actor "Registered User" as User
    actor "System Admin" as Admin
    
    package "Product Recommendation System" {
        usecase "Browse Products" as UC1
        usecase "Search Products" as UC2
        usecase "View Product Details" as UC3
        usecase "Get Personalized\nRecommendations" as UC4
        usecase "Login/Signup" as UC5
        usecase "View Dashboard" as UC6
        usecase "Manage Data Pipeline" as UC7
        usecase "Retrain Models" as UC8
    }

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6

    Admin --> UC5
    Admin --> UC7
    Admin --> UC8
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
    
    par Parallel Score Calculation
        API->>KNN: Predict Ratings (Collaborative)
        API->>TFIDF: Calculate Cosine Similarity (Content)
    end
    
    API->>API: Weighted Combine: Score = α·CF + (1-α)·Content
    
    API-->>U: Show Top 6 Hybrid Recommendations
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
