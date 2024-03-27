from flask import Flask,request,render_template
from src.mlproject.pipelines.predction_pipeline import PredictPipeline, CustomData




app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        data=CustomData(
            gender=request.form.get("gender"),
            age=float(request.form.get("age")),
            hypertension=float(request.form.get("hypertension")),
            heart_disease=float(request.form.get("heart_disease")),
            ever_married=request.form.get("ever_married"),
            work_type=request.form.get("work_type"),
            Residence_type=request.form.get("Residence_type"),
            avg_glucose_level=float(request.form.get("avg_glucose_level")),
            bmi=float(request.form.get("bmi")),
            smoking_status=request.form.get("smoking_status")
        )
        final_data=data.get_data_as_dataframe()

        predict_pipeline=PredictPipeline()

        pred=predict_pipeline.predict(final_data)



        return render_template("result.html",final_result=pred)



if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)