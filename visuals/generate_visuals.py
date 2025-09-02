import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------------------
# 1. Dummy Data (replace with real SpaceX dataset later)
# ------------------------------------------------------
data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "Launches": [6, 8, 18, 21, 13, 26, 31],
    "Success": [5, 8, 17, 20, 13, 25, 30],
    "Site": ["CCAFS", "KSC", "VAFB", "CCAFS", "KSC", "VAFB", "CCAFS"]
}
df = pd.DataFrame(data)

# ------------------------------------------------------
# 1. Launch Outcomes Over Time
# ------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["Year"], df["Launches"], marker="o", label="Total Launches")
plt.plot(df["Year"], df["Success"], marker="s", label="Successful Launches")
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Launch Outcomes Over Time")
plt.legend()
plt.savefig("visuals/launch_outcomes.png", dpi=300)
plt.close()

# ------------------------------------------------------
# 2. Launch Sites Success Rate
# ------------------------------------------------------
site_success = df.groupby("Site")["Success"].sum().reset_index()
plt.figure(figsize=(6, 5))
sns.barplot(data=site_success, x="Site", y="Success", palette="viridis")
plt.title("Success Rate by Launch Site")
plt.ylabel("Successful Launches")
plt.savefig("visuals/launch_sites.png", dpi=300)
plt.close()

# ------------------------------------------------------
# 3. Feature Importance (Dummy ML Model Example)
# ------------------------------------------------------
# Generate dummy feature data
X = pd.DataFrame({
    "PayloadMass": [3000, 5000, 4000, 6000, 3500, 4200, 5100],
    "BoosterReuse": [1, 0, 1, 1, 0, 1, 0],
    "OrbitType": [0, 1, 0, 1, 1, 0, 1]
})
y = [1, 0, 1, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Plot feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features, palette="coolwarm")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.savefig("visuals/feature_importance.png", dpi=300)
plt.close()

# ------------------------------------------------------
# 4. Model Accuracy Comparison
# ------------------------------------------------------
model_accuracies = {
    "Logistic Regression": 0.846,
    "SVM": 0.848,
    "Decision Tree": 0.876,
    "KNN": 0.848
}

plt.figure(figsize=(7, 5))
sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette="magma")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 0.9)
plt.xticks(rotation=20)
plt.savefig("visuals/model_accuracy.png", dpi=300)
plt.close()

print("✅ All plots generated and saved in 'visuals/' folder.")
