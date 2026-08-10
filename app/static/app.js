/**
 * This JS script contains all logic to handle API calls.
 */

// Get reference to DOM elements
const genButton = document.getElementById("StartNetworkInferenceButton");
const statusField = document.getElementById("StatusField");
const sourceChart = document.getElementById("sourceChart");
const genChart = document.getElementById("genChart");

/**
 * Requests generated samples from the API.
 * @param {number} numSamples Number of samples to generate.
 * @returns {Promise<any>} Promise resolving to the generated samples.
 */
async function requestSamples(numSamples) {
    const response = await fetch("/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({"num_samples": numSamples}),
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
    genButton.disabled = true;      // Disables the button for the processing time
    statusField.textContent = "Generating ...";

    try {
        const data = await requestSamples(200);  // Make the number of samples variable later!
        statusField.textContent = `Received ${data.num_samples} samples.`;
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