const form = document.querySelector("#prediction-form");
const button = document.querySelector("#predict-button");
const message = document.querySelector("#form-message");
const probability = document.querySelector("#probability");
const resultLabel = document.querySelector("#result-label");
const meter = document.querySelector(".meter");
const meterFill = document.querySelector("#meter-fill");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "";

  if (!form.reportValidity()) return;

  const values = new FormData(form);
  const payload = {
    age: Number(values.get("age")),
    annual_income: Number(values.get("annual_income")),
    annual_premium: Number(values.get("annual_premium")),
    vehicle_age_years: Number(values.get("vehicle_age_years")),
    previously_insured: form.elements.previously_insured.checked,
    vehicle_damage: form.elements.vehicle_damage.checked,
    region: values.get("region"),
  };

  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Calculating...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(data.detail || "Prediction request failed.");

    const percentage = Math.round(data.probability * 100);
    probability.innerHTML = `${percentage}<span>%</span>`;
    resultLabel.textContent = data.result;
    meterFill.style.width = `${percentage}%`;
    meter.setAttribute("aria-valuenow", String(percentage));
  } catch (error) {
    message.textContent =
      error.message || "Could not reach the prediction service.";
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent =
      "Estimate likelihood";
  }
});
