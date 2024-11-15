## How to run app interface
* Creare virtual environment

```python -m venv .venv ```  
```source .venv/bin/activate ```

* Clone the repo
  
```git clone https://github.com/gokul-gopan-k/Air_quality_predictor.git```

```cd Air_quality_predictor```

* Run requirements file

```pip install -r requirements.txt```

* Run the app
  
```python app.py```

# Problem statement

Current air quality monitoring systems are based on models that assume data exchangeability
which means that data points are interchangeable regardless of the time when they are taken.
This assumption does not hold as air quality in an area is influenced by a combination of factors
such as local emissions and the topography of the place, making the air quality dynamic.
Ignoring this assumption results in unreliable predictions which lead to misinformed decisions
and reduces public trust. Hence there is an immediate need for models that provide robust
uncertainty quantification as air pollution remains a challenge in the world. The World Health
Organization estimates that 4.2 million premature deaths occur annually due to outdoor air
pollution. The problem of air quality models has a global effect, ranging from urban to rural
centres. Hence addressing this challenge is crucial to ensure timely data-based decisions are
taken. Research to develop conformal risk control methods to tackle data non-exchangeability
can help move towards informed decision-making and safeguard against environmental
challenges.
