const form = document.getElementById("predictionForm");

const result = document.getElementById("result");
const prediction = document.getElementById("prediction");

const error = document.getElementById("error");
const errorMessage = document.getElementById("errorMessage");

const button = document.getElementById("predictButton");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    result.classList.add("hidden");
    error.classList.add("hidden");

    button.disabled = true;
    button.textContent = "Predicting...";


    const data = {
        year: parseFloat(document.getElementById("year").value),

        region: document.getElementById("region").value,

        manufacturer: document.getElementById("manufacturer").value,

        model: document.getElementById("model").value,

        condition: document.getElementById("condition").value || null,

        cylinders: document.getElementById("cylinders").value || null,

        fuel: document.getElementById("fuel").value || null,

        odometer: parseFloat(
            document.getElementById("odometer").value
        ),

        title_status:
            document.getElementById("title_status").value || null,

        transmission:
            document.getElementById("transmission").value || null,

        drive:
            document.getElementById("drive").value || null,

        type:
            document.getElementById("type").value || null,

        paint_color:
            document.getElementById("paint_color").value || null,

        state: document.getElementById("state").value,

        lat: parseFloat(
            document.getElementById("lat").value
        ),

        long: parseFloat(
            document.getElementById("long").value
        )
    };


    try {

        const response = await fetch("/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)
        });


        const responseData = await response.json();


        if (!response.ok) {

            let message = "Prediction failed.";

            if (Array.isArray(responseData.detail)) {

                message = responseData.detail
                    .map(error => {
                        return error.msg;
                    })
                    .join(", ");

            } else if (responseData.detail) {

                message = responseData.detail;
            }

            throw new Error(message);
        }


        const price = responseData.predicted_price;


        prediction.textContent =
            "$" + price.toLocaleString(
                "en-US",
                {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }
            );


        result.classList.remove("hidden");

    }


    catch (err) {

        console.error("Prediction error:", err);

        errorMessage.textContent =
            err.message ||
            "An unexpected error occurred.";

        error.classList.remove("hidden");
    }


    finally {

        button.disabled = false;
        button.textContent = "Predict Price";
    }

});