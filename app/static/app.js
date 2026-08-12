/**
 * This JS script contains all logic to handle API calls.
 */

// Get reference to DOM elements
const genButton = document.getElementById("StartNetworkInferenceButton");
const statusField = document.getElementById("StatusField");
const sourceChart = document.getElementById("sourceChart");
const genChart = document.getElementById("genChart");
const trajectoryChart = document.getElementById("trajectoryChart");
const numSamplesInput = document.getElementById("NumSamplesInput");
const integrationStepsInput = document.getElementById("IntegrationStepsInput");
const replayButton = document.getElementById("ReplayTrajectoryButton");

// Frame names of the most recently built trajectory animation.
let trajectoryFrameNames = [];

// Shared playback settings.
const TRAJECTORY_PLAYBACK = {
    frame: { duration: 30, redraw: false },
    transition: { duration: 0 },
    mode: "immediate",
};


// --------------- FUNCTIONS ---------------

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
            "return_trajectory": true,  // TODO: Make this configurable in the frontend, e.g., via a checkbox. Maybe make this fixed after all? Would be less complicated for a demo.
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
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: getComputedStyle(document.body).color },
        title: {text: title}
    };
    Plotly.react(container, plotData, layout, {responsive: true});
}

/**
 * Plays the trajectory animation from its first frame.
 * Frames are already registered on the plot, so replaying needs no new request.
 * @param {HTMLElement} container HTML element holding the animation.
 * @returns {Promise<void>} Resolves once the animation has finished.
 */
async function playTrajectory(container) {
    if (trajectoryFrameNames.length === 0) {
        return;
    }

    replayButton.disabled = true;
    try {
        await Plotly.animate(container, trajectoryFrameNames, TRAJECTORY_PLAYBACK);
    } catch (error) {
        // Plotly rejects when an animation is interrupted; not worth surfacing.
        console.error(error);
    } finally {
        replayButton.disabled = false;
    }
}

/**
 * Builds a point cloud animation along its ODE trajectory and plays it once.
 * @param {HTMLElement} container HTML element to contain the animation.
 * @param {number [][][]} trajectory Points per step, shape (steps, samples, 2).
 * @param {string} title Title of the plot.
 * @returns {Promise<void>} Resolves once the first playthrough has finished.
 */
function animateTrajectory(container, trajectory, title) {

    // Transpose once, upfront — never inside the animation loop.
    const frames = trajectory.map((points, i) => {
        const [x, y] = transpose(points);
        return { name: String(i), data: [{ x, y }] };
    });

    const layout = {
        xaxis: { autorange: false, range: [-3, 3] },
        yaxis: { autorange: false, range: [-3, 3], scaleanchor: "x" },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: getComputedStyle(document.body).color },
        title: { text: title },
    };

    const trace = {
        ...frames[0].data[0],
        mode: "markers",
        type: "scatter",
        marker: { color: "rgb(176, 114, 214)", size: 4, opacity: 0.5 },
    };

    // newPlot (not react) resets any frames left over from a previous run.
    Plotly.newPlot(container, [trace], layout, { responsive: true });
    Plotly.addFrames(container, frames);

    // Remember the frame names so the replay button can re-run them later.
    trajectoryFrameNames = frames.map(f => f.name);

    return playTrajectory(container);
}


// --------------- EVENT LISTENERS ---------------

/**
 * Handels the click event for the "Generate Samples" button.
 * Validates the input fields, sends a request to the API, and plots the received samples.
 */
genButton.addEventListener("click", async function () {
    if (!numSamplesInput.reportValidity() || !integrationStepsInput.reportValidity()) {
        return; // Exit if inputs are invalid
    }

    genButton.disabled = true;      // Disables the button for the processing time
    replayButton.disabled = true;   // The current animation is about to be replaced
    statusField.textContent = "Generating ...";

    try {
        const data = await requestSamples(
            Number(numSamplesInput.value),
            Number(integrationStepsInput.value),
        );
        statusField.textContent = `Received ${data.num_samples} samples.`;
        console.log(data);

        // Plot point clouds for source and generated distributions and animate the trajectory.
        plotPointCloud(sourceChart, data.source_points, "Source distribution");
        plotPointCloud(genChart, data.generated_points.at(-1), "Generated distribution");
        animateTrajectory(trajectoryChart, data.generated_points, "Trajectory animation");
    } catch (error) {
        statusField.textContent = `Error: ${error.message}`;
        console.error(error);

        // Keep replay usable if an earlier animation is still on screen.
        replayButton.disabled = trajectoryFrameNames.length === 0;
    } finally {
        genButton.disabled = false;
    }
});

/**
 * Handles click event for the "Replay" button.
 * Just reruns the latest generated trajectory w/ frames saved in variable "trajectoryFrameNames".
 */
replayButton.addEventListener("click", function () {
    playTrajectory(trajectoryChart);
});