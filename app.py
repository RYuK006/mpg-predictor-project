import gradio as gr
import pandas as pd
import joblib
import os

os.environ["MPLCONFIGDIR"] = "/tmp"
reg_model = joblib.load("best_vehicle_mpg_model.pkl")

makes = ['FORD', 'TOYOTA', 'HONDA', 'BMW', 'AUDI']
transmissions = ['Automatic', 'Manual', 'CVT', 'Auto-Manual', 'Direct Drive']
fuels = ['Regular Gasoline', 'Premium Gasoline', 'Diesel', 'E85', 'CNG']
classes = ['SUV', 'Compact Cars', 'Mid-size Cars', 'Large Cars', 'Pickup Trucks', 'Station Wagons']

def predict_mpg(year, cylinders, displ, make, trany, fuel, vclass):
    input_data = pd.DataFrame([{
        "year": year,
        "cylinders": cylinders,
        "displ": displ,
        "make": make,
        "trany": trany,
        "fuelType": fuel,
        "VClass": vclass
    }])
    prediction = reg_model.predict(input_data)[0]
    return f"🚗 Predicted Combined MPG: {prediction:.2f} MPG"

profile_cards = """
<div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
  <div class="profile-card">
    <img src="https://huggingface.co/spaces/Aaron006/Aaron/resolve/main/Picsart_25-04-15_20-22-25-802.jpg" class="avatar" />
    <h2 class="name">Aaron</h2>
    <p class="title">Machine Learning Explorer</p>
    <a href="https://www.linkedin.com/in/aaron-alex-mathew/" target="_blank" class="linkedin-btn">💼 Connect on LinkedIn</a>
  </div>
  <div class="profile-card">
    <img src="https://huggingface.co/spaces/Aaron006/Aaron/resolve/main/1723800318684.jpg" class="avatar" />
    <h2 class="name">Agnivesh P A</h2>
    <p class="title">Machine Learning Explorer</p>
    <a href="https://www.linkedin.com/in/agnivesh-p-a-4747a1258/" target="_blank" class="linkedin-btn">💼 Connect on LinkedIn</a>
  </div>
</div>
"""

built_with = """
<div class="built-with">
  <h3>🔧 Built With:</h3>
  <div class="badges">
    <span class="badge">Python</span>
    <span class="badge">scikit-learn</span>
    <span class="badge">Gradio</span>
  </div>
</div>
"""

custom_css = """
body { background: #0f0f1a; font-family: sans-serif; }

.profile-card {
  width: 280px;
  height: 380px;
  text-align: center;
  padding: 20px;
  margin: 20px;
  border-radius: 20px;
  background: linear-gradient(145deg, #2c2f4a, #1b1c30);
  box-shadow: 0 0 30px rgba(0,255,255,0.2);
  transition: transform 0.3s ease-in-out;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.profile-card:hover {
  transform: scale(1.05);
  box-shadow: 0 0 40px rgba(0,255,255,0.4);
}

.avatar {
  width: 120px;
  height: 120px;
  object-fit: cover;
  border-radius: 50%;
  border: 3px solid rgba(255,255,255,0.2);
  margin-bottom: 10px;
  align-self: center;
}

.name { font-size: 22px; color: #ffffff; margin: 6px 0; }
.title { font-size: 14px; color: #ccc; margin-bottom: 16px; }

.linkedin-btn {
  padding: 10px 16px;
  background-color: #0077b5;
  color: white;
  border-radius: 8px;
  text-decoration: none;
  font-weight: bold;
}
.linkedin-btn:hover { background-color: #005580; }

.built-with {
  text-align: center;
  margin-top: 10px;
  margin-bottom: 30px;
  color: #ccc;
}

.built-with h3 {
  font-size: 18px;
  color: #fff;
  margin-bottom: 10px;
}

.badges {
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.badge {
  padding: 6px 12px;
  background-color: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  font-size: 13px;
  color: #fff;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(profile_cards)
    gr.HTML(built_with)

    gr.Markdown("## 🚘 Vehicle MPG Predictor")
    gr.Markdown("Enter vehicle details to predict Combined MPG")

    with gr.Row():
        with gr.Column():
            year = gr.Slider(2000, 2025, value=2020, step=1, label="Year")
            cylinders = gr.Slider(2, 16, value=4, step=1, label="Cylinders")
            displ = gr.Slider(1.0, 8.0, value=2.0, step=0.1, label="Engine Displacement (L)")
            trany = gr.Dropdown(transmissions, label="Transmission")
        with gr.Column():
            make = gr.Dropdown(makes, label="Make")
            fuel = gr.Dropdown(fuels, label="Fuel Type")
            vclass = gr.Dropdown(classes, label="Vehicle Class")

    result = gr.Textbox(label="Prediction Result")
    predict_btn = gr.Button("🚀 Predict MPG")

    predict_btn.click(
        fn=predict_mpg,
        inputs=[year, cylinders, displ, make, trany, fuel, vclass],
        outputs=result
    )

demo.launch()
