# IBM AI Enterprise Workflow Capstone

The project description can be found [here](https://github.com/aavail/ai-workflow-capstone).

## Part 1: Data Investigation

``` ~$ python data_ingestion.py ```

``` ~$ python data_visualization.py ```

## Part 2: Model Building & Selection

``` ~$ python feature_engineering.py ```

``` ~$ python logger.py

``` ~$ python model.py ```

## Part 3: Model Production

``` ~$ python app.py ```

``` ~$ docker build -t capstone-ai-app . ```

``` ~$ docker run -p 4000:8080 capstone-ai-app ```
