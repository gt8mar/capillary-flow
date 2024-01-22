# import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns

# # Assuming df is your DataFrame
# df = pd.read_csv('C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\velocities\\big_df.csv') # Load your data

# # Scatter plot for Age vs Velocity
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Pressure', y='Velocity',hue = 'SYS_BP', data=df)
# # Color points by age

# plt.title('Pressure vs Velocity')
# plt.xlabel('Pressure')
# plt.ylabel('Velocity')
# plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# # Load data
# # df = pd.read_csv('your_data.csv')
# df = pd.read_csv('C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\velocities\\big_df.csv') # Load your data


# # Standardizing the features
# features = ['Velocity', 'Pressure', 'SYS_BP']
# x = df.loc[:, features].values
# x = StandardScaler().fit_transform(x)

# # PCA
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)

# # Create a DataFrame with the PCA results
# principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
# principalDf['Age'] = df['Age']  # Add the age column

# # Plotting
# plt.figure(figsize=(10,8))
# sns.scatterplot(x='principal component 1', y='principal component 2', hue='Age', data=principalDf, palette='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('2 Component PCA Colored by Age')
# plt.colorbar(scatter,label='Age')
# plt.show()