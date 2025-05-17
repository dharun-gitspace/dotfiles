## **Problem statement** : 
Many individuals struggle with mental health issues but hesitate to take traditional assessments due to stigma or lack of engagement. Existing methods rely on direct questionnaires, leading to biased responses and reduced accuracy. This application offers a **story-driven assessment**, where users interact with real-life scenarios, making the evaluation more natural and engaging. Using **machine learning and data mining techniques**, the system accurately detects disorders and provides **personalized suggestions**, including lifestyle changes, specialist recommendations, and educational resources, ensuring a comprehensive and supportive mental health solution.
## **1. Scope**

The Mental Health Assessment App is designed to provide users with an engaging and story-driven approach to mental health evaluation. By analyzing user responses through interactive scenarios and direct data inputs, the app determines the presence of mental health conditions and provides personalized recommendations, specialist referrals, and educational resources. The system integrates **data mining techniques and machine learning models** to enhance assessment accuracy.

### **1.1 In Scope**

- Story-driven mental health assessments across various levels (e.g., Sleep, Stress, Anxiety, etc.).
- Data collection through **interactive choices and direct responses** (e.g., sleep hours, stress levels).
- Machine learning-based disorder detection and evaluation.
- Personalized recommendations, including lifestyle changes, therapy suggestions, and reading materials.
- Secure user data storage and compliance with **GDPR and HIPAA** regulations.

### **1.2 Out of Scope**

- Providing direct medical diagnosis or replacing clinical psychiatric evaluations.
- Offering real-time therapy or direct communication with professionals.
- Social interaction features such as chat rooms or community forums.
- Integration with third-party medical record systems.

---

## **2. Primary Objectives**

- **Provide a structured assessment experience** using **story-driven scenarios** rather than traditional surveys.
- **Detect mental health conditions** (e.g., anxiety, depression, sleep disorders) based on user responses.
- **Leverage data mining and machine learning** to improve assessment accuracy over time.
- **Deliver actionable recommendations** to help users improve their mental well-being.
- **Ensure privacy and security** by encrypting user data and following compliance standards.

### **2.1 Key Performance Indicators (KPIs)**

- **Assessment Completion Rate** – Percentage of users completing all levels.
- **Accuracy of Disorder Detection** – Comparison with validated psychological assessments.
- **User Engagement** – Time spent per level and completion rates.
- **Recommendation Effectiveness** – Measured through user feedback.
- **System Performance** – Latency in loading assessments and generating reports.

---

## **3. Functional Requirements**

### **3.1 Data Collection**

- Gather responses from interactive **story-based scenarios**.
- Collect direct mental health data (e.g., sleep hours, stress levels, exercise habits).
- Store user assessment history for tracking progress over time.

### **3.2 Data Analysis**

- Use machine learning models to analyze responses and detect mental health patterns.
- Compare user inputs with benchmark datasets for accuracy.
- Generate reports on **user mental state and risk factors**.

### **3.3 Recommendation System**

- Provide personalized recommendations based on assessment results.
- Suggest **specialist referrals** (e.g., psychologists, therapists) if required.
- Offer **lifestyle changes, dietary advice, and mental well-being practices**.
- Curate related articles and resources based on detected conditions.

### **3.4 Visualization and Reporting**

- Display graphical insights on mental health trends.
- Generate downloadable reports (PDF, CSV) for user reference.
- Provide progress tracking for repeated assessments.

### **3.5 Security and Compliance**

- Encrypt all stored user data and ensure secure authentication.
- Implement role-based access control (RBAC) for user profiles.
- Adhere to **GDPR, HIPAA, and data privacy regulations**.
- Provide user consent mechanisms for data collection and storage.

---

## **4. Non-Functional Requirements**

### **4.1 Performance**

- The app should process assessments **within 3 seconds** per response.
- Report generation should be completed **instantly after assessment completion**.

### **4.2 Maintainability**

- The system should support **regular updates** to refine machine learning models.
- Ensure easy integration of new mental health levels and scenarios.

### **4.3 Accessibility**

- Provide **a user-friendly UI with calming themes** to reduce distress.
- Support **voice-over features and text-to-speech capabilities**.

### **4.4 Interoperability**

- Enable integration with **wearable devices** (e.g., sleep trackers, fitness apps) for better data collection.
- Allow API integration with third-party **mental health research tools**.

### **4.5 Reliability**

- Maintain **99% uptime** for assessments and report generation.
- Implement **backup and recovery mechanisms** to prevent data loss.

---

## **5. Use Case Diagram**

### **Actors**

- **User**: Takes assessments, receives recommendations, and tracks progress.
- **System AI**: Analyzes responses, applies machine learning models, and generates reports.
- **Suggestion system**: Provide suggestions and help like which specialist to seek, articles.

### **Use Cases**

1. **Start Assessment** – User begins a mental health evaluation.
2. **Progress Through Story-Based Levels** – User interacts with story-driven scenarios.
3. **Answer Questions** – Users respond to different situations presented in levels.
4. **Provide Direct Data** – Users enter factual data (e.g., sleep hours, stress levels).
5. **Analyze Responses** – System AI processes inputs and detects patterns.
6. **Determine Mental Health Condition** – AI classifies user responses and detects potential disorders.
7. **Provide Recommendations** – System offers therapy suggestions, lifestyle changes, and relevant articles.
8. **Allow Retaking Assessments** – Users can reattempt levels to track mental health progress.
### **Use Case Diagram** : 
![[usecase diagram.png]]
## **6. Extended Use Case**

| **Use Case Name**                       | Detecting Mental Health Disorder                                                                                                                                                                                                                                                                                        |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Scope**                               | Mental Health Assessment App                                                                                                                                                                                                                                                                                            |
| **Level**                               | User Goal                                                                                                                                                                                                                                                                                                               |
| **Primary Actor Supporting Actors**     | User<br>SystemAI, Suggestion system<br>                                                                                                                                                                                                                                                                                 |
| **Stakeholders and Interests**          | - **User**: Wants accurate insights into mental health. - **Mental Health Specialist**: Requires reliable reports for potential follow-ups. - **System AI**: Needs accurate data for improved predictions.                                                                                                              |
| **Preconditions**                       | - The user must complete at least one assessment level. - The system has enough training data for accurate predictions.                                                                                                                                                                                                 |
| **Success Guarantee**                   | - The system provides a **clear assessment report**. - User receives **relevant mental health recommendations**.                                                                                                                                                                                                        |
| **Main Success Scenario**               | 1. User completes an interactive mental health assessment. 2. System AI processes responses and applies data mining techniques. 3. Machine learning model **identifies mental health risk factors**. 4. System generates a **personalized mental health report**. 5. User receives **recommendations for improvement**. |
| **Extensions**                          | - If user data is **incomplete**, system prompts them to provide missing inputs. - If a serious mental health concern is detected, the system **recommends urgent specialist consultation**.                                                                                                                            |
| **Technology and Data Variations List** | - Utilizes **machine learning models** trained on mental health datasets. - Supports **text-based and graphical user interfaces** for assessment delivery. - **Stores encrypted user responses** in a secure database.                                                                                                  |
| **Frequency of Occurrence**             | - Typically used **once per assessment session**. - Users may **reassess periodically** to track mental health trends. - System updates models **regularly** for improved accuracy.                                                                                                                                     |
| **Miscellaneous**                       | - Provides **anonymized analytics** for research purposes. - Future iterations may include **wearable device integration** for real-time data tracking. - Ensures compliance with **GDPR and HIPAA** for data security.                                                                                                 |
