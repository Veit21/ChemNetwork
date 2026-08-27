/**
 * This JS script contains all logic to handle API calls.
 */

// Get reference to DOM elements
const parameterForm = document.getElementById("ParameterForm");
const genButton = document.getElementById("StartNetworkInferenceButton");
const statusField = document.getElementById("StatusField");
const sourceChart = document.getElementById("sourceChart");
const genChart = document.getElementById("genChart");
const trajectoryChart = document.getElementById("trajectoryChart");
const numSamplesInput = document.getElementById("NumSamplesInput");
const integrationStepsInput = document.getElementById("IntegrationStepsInput");
const targetDistributionDropdown = document.getElementById("TargetDistributionInput")
const replayButton = document.getElementById("ReplayTrajectoryButton");

// Frame names of the most recently built trajectory animation.
let trajectoryFrameNames = [];

// Shared playback settings.
const TRAJECTORY_PLAYBACK = {
    frame: { duration: 30, redraw: false },
    transition: { duration: 0 },
    mode: "immediate",
};

// TODO: Make a struct-like thing here to define plotting parameters s.a. point size/opacity/plot range etc.


// --------------- FUNCTIONS ---------------

/**
 * Requests the available target distributions from the API endpoint.
 * @returns Available targets.
 */
async function requestAvailableTargets() {
    const response = await fetch("/available", {
        method: "GET",
        headers: {"Content-Type": "application/json"},
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
}

/**
 * Populates the target dropdown from the API and preselects the server's default.
 * @returns {Promise<void>} Resolves once the dropdown is filled or the error is shown.
 */
async function init() {
    try {
        const {targets, default: defaultTarget} = await requestAvailableTargets();
        targetDistributionDropdown.append(
            ...targets.map(target => new Option(target.label, target.id))
        );
        targetDistributionDropdown.value = defaultTarget;
    } catch (error) {
        statusField.textContent = `Could not load target distributions: ${error.message}`;
        genButton.disabled = true;
        console.error(error);
    }
}

/**
 * Requests generated samples from the API.
 * @param {number} numSamples Number of samples to generate.
 * @param {number} integrationSteps Number of integration steps.
 * @param {string} targetDistribution Target distribution to generate samples from.
 * @returns {Promise<any>}} Promise resolving to the generated samples.
 */
async function requestSamples(numSamples, integrationSteps, targetDistribution) {
    const response = await fetch("/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            "num_samples": numSamples,
            "integration_steps": integrationSteps,
            "target": targetDistribution,
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
    const plotData = {
        x: pointsTransposed[0],
        y: pointsTransposed[1],
        mode: "markers",
        type: "scatter",
        marker: {
            color: "rgb(176, 114, 214)",
            size: 4,
            opacity: 0.5,
        },
    };
    const layout = {
        yaxis: {
            autorange: false,
            range: [-4, 4],
            scaleanchor: "x",
        },
        xaxis: {
            autorange: false,
            range: [-4, 4],
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: getComputedStyle(document.body).color },
        title: {text: title}
    };
    Plotly.react(container, [plotData], layout, {responsive: true});
}

/**
 * Plots both the generated and target point cloud for showing the result and make it visibly comparable.
 * @param {HTMLElement} container HTML element to contain the plot. 
 * @param {number[][]} points_set_1 First set of points to plot, i.e. data points drawn from the learend distribution.
 * @param {number[][]} points_set_2 Second set of points to plot, i.e. the ground truth target distribution.
 * @param {string} title Title of the plot.
 */
function plotPointClouds(container, points_set_1, points_set_2, title) {
    const pointsSet1Transposed = transpose(points_set_1);
    const pointsSet2Transposed = transpose(points_set_2);
    
    const plotDataSet1 = {
        x: pointsSet1Transposed[0],
        y: pointsSet1Transposed[1],
        mode: "markers",
        type: "scatter",
        name: "Generated",
        marker: {
            color: "rgb(176, 114, 214)",
            size: 4,
            opacity: 0.5,
        },
    };

    const plotDataSet2 = {
        x: pointsSet2Transposed[0],
        y: pointsSet2Transposed[1],
        mode: "markers",
        type: "scatter",
        name: "Target",
        marker: {
            color: "rgb(130, 190, 85)",
            size: 4,
            opacity: 0.5,
        },
    };
    
    const layout = {
        yaxis: {
            autorange: false,
            range: [-4, 4],
            scaleanchor: "x",
        },
        xaxis: {
            autorange: false,
            range: [-4, 4],
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: getComputedStyle(document.body).color },
        title: {text: title}
    };
    
    Plotly.react(container, [plotDataSet1, plotDataSet2], layout, {responsive: true});
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
        xaxis: { autorange: false, range: [-4, 4] },
        yaxis: { autorange: false, range: [-4, 4], scaleanchor: "x" },
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
 * Handles the submit event of the parameter form.
 * Sends a request to the API and plots the received samples.
 */
parameterForm.addEventListener("submit", async function (event) {
    event.preventDefault();         // Results are fetched
    genButton.disabled = true;      // Disables the button for the processing time
    replayButton.disabled = true;   // The current animation is about to be replaced
    statusField.textContent = "Generating ...";

    try {
        const data = await requestSamples(
            Number(numSamplesInput.value),
            Number(integrationStepsInput.value),
            targetDistributionDropdown.value,
        );
        statusField.textContent = `Received ${data.num_samples} samples.`;
        console.log(data);

        // Plot point clouds for source and generated distributions and animate the trajectory.
        plotPointCloud(sourceChart, data.source_points, "Source distribution");
        plotPointClouds(genChart, data.generated_points.at(-1), data.target_points, "Generated distribution");
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


// --------------- INITIALISATION ---------------

init();
