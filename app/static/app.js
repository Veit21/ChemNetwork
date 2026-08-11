/**
 * This JS script contains all logic to handle API calls.
 */

// Get reference to DOM elements
const genButton = document.getElementById("StartNetworkInferenceButton");
const statusField = document.getElementById("StatusField");
const sourceChart = document.getElementById("sourceChart");
const genChart = document.getElementById("genChart");
const numSamplesInput = document.getElementById("NumSamplesInput");
const integrationStepsInput = document.getElementById("IntegrationStepsInput");

/**
 * Requests generated samples from the API.
 * @param {number} numSamples Number of samples to generate.
 * @param {number} integrationSteps Number of integration steps.
 * @returns {Promise<any>}} Promise resolving to the generated samples.
 */
async function requestSamples(numSamples, integrationSteps) {
    const response = await fetch("/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            "num_samples": numSamples,
            "integration_steps": integrationSteps,
        }),
    });

    if (!response.ok) {
        const payload = await response.json().catch(() => null); // What does this mean?
        throw new Error(`HTTP ${response.status}: ${JSON.stringify(payload?.detail ?? "unknown error")}`); // ...and this?
    }

    return response.json();
}

/**
 * Transposes a N x M matrix.
 * @param {number[][]} matrix Input matrix to be transposed.
 * @returns {number[][]} Transposed matrix.
 */
function transpose(matrix) {
    return matrix[0].map((_, c) => matrix.map(row => row[c]));
}

/**
 * Plots a point cloud using Plotly.js.
 * @param {HTMLElement} container HTML element to contain the plot.
 * @param {number[][]} points Data points to be plotted, expected as an array of [x, y] pairs.
 * @param {string} title Title of the plot.
 */
function plotPointCloud(container, points, title) {
    const pointsTransposed = transpose(points);
    const plotData = [{
        x: pointsTransposed[0],
        y: pointsTransposed[1],
        mode: "markers",
        type: "scatter",
        marker: {
            color: "rgb(176, 114, 214)",
            size: 4,
            opacity: 0.5,
        },
    }];
    const layout = {
        yaxis: {
            autorange: false,
            range: [-3, 3],
            scaleanchor: "x",
        },
        xaxis: {
            autorange: false,
            range: [-3, 3],
        },
        title: {text: title}};
    Plotly.react(container, plotData, layout);
}

genButton.addEventListener("click", async function () {
    if (!numSamplesInput.reportValidity() || !integrationStepsInput.reportValidity()) {
        return; // Exit if inputs are invalid
    }

    genButton.disabled = true;      // Disables the button for the processing time
    statusField.textContent = "Generating ...";

    try {
        const data = await requestSamples(
            Number(numSamplesInput.value),
            Number(integrationStepsInput.value),
        );
        statusField.textContent = `Received ${data.num_samples} samples.`;
        console.log(numSamplesInput.value);
        console.log(data);

        // Plot point clouds for source and generated distributions
        plotPointCloud(sourceChart, data.source_points, "Source distribution");
        plotPointCloud(genChart, data.generated_points, "Generated distribution");
    } catch (error) {
        statusField.textContent = `Error: ${error.message}`;
        console.error(error);
    } finally {
        genButton.disabled = false;
    } 
});