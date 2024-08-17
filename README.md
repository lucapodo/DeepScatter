
# DeepScatter: Visual Approach for Shift Anomaly Enhancing in Time Series Data

Understanding and detecting shift anomalies in time series data is a complex task, as these anomalies persistently deviate from the typical trends. Unlike isolated point anomalies, shift anomalies disrupt the overall pattern, making their identification crucial across various domains. 

**DeepScatter** is an innovative package that combines a straightforward deep learning model, specifically an autoencoder, with scatter plots to enhance the visualization and detection of shift anomalies. The power of deep learning models is leveraged to capture complex patterns within the data, enabling a comprehensive understanding and identification of shift anomalies.

By integrating deep learning algorithms, visual analytics techniques, and domain expert decision processes, this package provides a robust solution. The compact architecture of the autoencoder effectively learns the latent features of time series data, making shift anomalies visually distinguishable in scatter plots.

## Repository Structure

- **docs/**: Contains the detailed documentation of the DeepScatter package.
- **experiments.pdf**: A study on the opportunities and limitations of the approach on various datasets.
- **deepscatter/**: Contains the source code for the DeepScatter package.
- **notebooks/**: 
  - `demo.ipynb`: A demo notebook that shows how to use the DeepScatter package.
  - `Deepscatter.ipynb`: A step-by-step explanation of the code and its functionality.

## How to Use

To use DeepScatter, follow these steps:

1. Clone the repository to your local machine:

   \`\`\`bash
   git clone https://github.com/yourusername/deepscatter.git
   \`\`\`

2. Change into the repository directory:

   \`\`\`bash
   cd deepscatter
   \`\`\`

3. Install the package using pip:

   \`\`\`bash
   pip install -e .
   \`\`\`

4. You can now use DeepScatter in your projects. Below is an example code snippet:

   \`\`\`python
   import tensorflow as tf
   from deepscatter.model import DeepScatter
   from deepscatterz import load_data, mark_anomalies_in_timeseries, plot_shit_anomaly, plotD, train_test_split

   df = load_data()
   df = df[0:3000]
   df = mark_anomalies_in_timeseries(df, "2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000")

   plot_shit_anomaly(df)

   train, test = train_test_split(df, 1700)
   dp = DeepScatter(train, test, verbose=False)
   dp.train_model()
   trian_res, test_res = dp.evaluate(5)

   plotD(trian_res, test_res, df)
   \`\`\`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

