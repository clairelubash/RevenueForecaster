# IBM AI Enterprise Workflow Capstone

A tool to predict revenue given a specific timeframe and market. The project description can be found [here](https://github.com/aavail/ai-workflow-capstone).

## Part 1: Data Investigation

Detailed insights for this section can be found in the notebook `Pt1.ipynb`.

### Data Ingestion

``` $ python data_ingestion.py ```

Converted time series data are saved as csv files in the *data* sub-directory.

### Data Visualization

``` $ python data_visualization.py ```

## Part 2: Model Building & Selection

Detailed insights for this section can be found in the notebook `Pt2.ipynb`.

### Feature Engineering

``` $ python feature_engineering.py ```

### Logfile Updates

``` $ python logger.py ```

The logs are saved as log files in the *logs* sub-directory.

### Modeling

``` $ python model.py ```

The models are saved as joblib files in the *models* sub-directory.

## Part 3: Model Production

Detailed insights for this section can be found in the notebook `Pt3.ipynb`.

### Test Flask API

``` $ python app.py ```

After starting the app, go to http://localhost:8080/.

### Unit Testing

The unit tests for the model and the API can be run through the following script:

``` $ run_tests.py ```

The code for the individual unit tests are in the *unit_tests* sub-directory.

### Docker Container

Build the image and then run the container.

``` $ docker build -t capstone-ai-app . ```

``` $ docker run -p 4000:8080 capstone-ai-app ```

To ensure app is running properly, go to http://localhost:4000/.
