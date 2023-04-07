

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts\\model.pkl"
            preprocessor_path = "artifacts\\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, month: str, visitortype: str, operatingsystems: int, browser: int, region: int, traffictype: int, weekend: int, administrative: float, administrative_duration: float, informational: float, informational_duration: float, productrelated: float, productrelated_duration: float, bouncerates: float, exitrates: float, pagevalues: float, specialday: float):
        self.month = month
        self.visitortype = visitortype
        self.operatingsystems = operatingsystems
        self.browser = browser
        self.region = region
        self.traffictype = traffictype
        self.weekend = weekend
        self.administrative = administrative
        self.administrative_duration = administrative_duration
        self.informational = informational
        self.informational_duration = informational_duration
        self.productrelated = productrelated
        self.productrelated_duration = productrelated_duration
        self.bouncerates = bouncerates
        self.exitrates = exitrates
        self.pagevalues = pagevalues
        self.specialday = specialday


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "month": [self.month],
            "visitortype": [self.visitortype],
            "operatingsystems": [self.operatingsystems],
            "browser": [self.browser],
            "region": [self.region],
            "traffictype": [self.traffictype],
            "weekend": [self.weekend],
            "administrative": [self.administrative],
            "administrative_duration": [self.administrative_duration],
            "informational": [self.informational],
            "informational_duration": [self.informational_duration],
            "productrelated": [self.productrelated],
            "productrelated_duration": [self.productrelated_duration],
            "bouncerates": [self.bouncerates],
            "exitrates": [self.exitrates],
            "pagevalues": [self.pagevalues],
            "specialday": [self.specialday],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            print(df)
            return df
        except Exception as e:
            raise CustomException(e,sys)

