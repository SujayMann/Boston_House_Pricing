## Boston_House_Pricing

### Software And Tools Required

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/)
3. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
4. [Render Account](https://dashboard.render.com)

Create a new environment
```
conda create -p venv python==3.10 -y
```
Add dependencies to requirements.txt

Create a Regression model ```Boston_Pricing.ipynb``` to use for the predictions

Pickle the model as ```regmodel.pkl``` to use in the flask app

```
pickle.dump()
```

Create a Flask app ```app.py```

Create a simple html page ```home.html``` to take inputs for the model

Provide the inputs to the model inside the flask app
* Use ```scaling.pkl``` to transform the input values
* Display the predicted value on the webpage

Run and test the app
```
python app.py
```

Deploy the app on [Render](dashboard.render.com)
