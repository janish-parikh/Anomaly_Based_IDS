# Anomaly Based Intrusion Detection System using UNSW_NB15 Dataset
UNSW_NB15 Dataset(all relevant features):

Workflow:

NB 15 dataset used, directly used the csv files available
Data Summary: 45% of training set is normal flows and rest is malicious
Features Used:  
cat_columns =   proto 
num_columns =   dur  ,    spkts  ,   dpkts  ,   sbytes  ,   dbytes  ,
  rate  ,   sttl  ,   dttl  ,   sload  ,   dload  ,   sloss  ,   dloss  ,   sinpkt  ,   dinpkt  ,   sjit  ,   djit  ,   swin  ,   stcpb  ,  dtcpb  ,   dwin  ,   tcprtt  ,   synack  ,   ackdat  ,   smean  ,   dmean  ,   trans_depth  ,   response_body_len  ,   ct_srv_src  ,  ct_state_ttl  ,   ct_dst_ltm  ,   ct_src_dport_ltm  ,   ct_dst_sport_ltm  ,   ct_dst_src_ltm  ,   is_ftp_login  ,   ct_ftp_cmd  ,   ct_flw_http_mthd  ,   ct_src_ltm  ,   ct_srv_dst  ,   is_sm_ips_ports  

Using these features and train data, defined a Spark ML Pipeline with the following stages:

- Stage 1-2: For each cat column : 2 stages: StringIndexer → OneHotEncoder
- Stage 3: All numerical columns and OneHotEncoder is transformed into a single features vector using VectorAssembler
- Stage 4: Final Stage is the Random Forest Classifier Model with hyperparameters: 

Random Forest Classifier Parameters
- No of trees: 150
- MaxDepth : 15

After fitting the pipeline on training data, the pipeline is persisted.

For testing the pipeline is loaded and we transform the test data using the fitted pipeline to obtain predictions and metrics to evaluate the model.
Metrics Obtained: accuracy is 0.972864687670734

                 precision    recall  f1-score   support

           0         0.96     0.98        0.97        7376
           1         0.98     0.97        0.98        9097

    accuracy                           0.97      16473
    macro avg      0.97      0.97      0.97      16473
    weighted avg   0.97      0.97      0.97      16473
