import * as tfjs from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.fesm.min.js";

export class FaceExample extends HTMLElement {
	static get observedAttributes() {
		return [];
	}
	constructor() {
		super();
		this.bind(this);
	}
	bind(element) {
		element.render = element.render.bind(element);
		element.attachEvents = element.attachEvents.bind(element);
		element.cacheDom = element.cacheDom.bind(element);
		element.startCamera = element.startCamera.bind(element);
		element.runPredictLoop = element.runPredictLoop.bind(element);
	}
	connectedCallback() {
		this.render();
		this.cacheDom();
		this.context = this.dom.canvas.getContext("2d");
		this.attachEvents();
	}
	render(){
		this.attachShadow({ mode: "open" });
		this.shadowRoot.innerHTML = `
			<style>
				#panel { display: grid; grid-template-columns: 50% 50%; grid-template-areas: "left right" }
				#video { grid-area: left; }
				#info { grid-area: right; }
			</style>
			<div id="panel">
				<video id="video" height="640" width="640"></video>
				<div id="info">
					<div id="output"></div>
					<canvas id="canvas" height="640" width="640"></canvas>
				</div>
			</div>
			<button id="start">Start</button>
			<button id="predict">Predict</button>
		`
	}
	async startCamera(){
		const userMediaPromise = navigator.mediaDevices.getUserMedia({ 
			video: true,
			height: 640, 
			width: 640 
		}).then(stream => {
			this.dom.video.srcObject = stream;
			this.dom.video.play();
			return new Promise((res, rej) => {
				this.dom.video.addEventListener("loadeddata", res);
			});
		});

		const modelPromise = tfjs.loadGraphModel(`./model/model.json`);

		const [_, model] = await Promise.all([userMediaPromise, modelPromise]);
		this.model = model;
		this.runPredictLoop();
	}
	async runPredictLoop(){
		const videoFrameTensor = tfjs.browser.fromPixels(this.dom.video);
		const resizedFrameTensor = tfjs.image.resizeBilinear(videoFrameTensor, [192, 192], true);
		const normalizedFrameTensor = resizedFrameTensor.div(255);

		const predictions = await this.model.predict(normalizedFrameTensor.expandDims());
		const faceDetection = await predictions[1].data();
		const isFace = faceDetection[0] > 0.8;
		const mesh = await predictions[0].data();

		this.dom.output.textContent = isFace ? `Face Found (${faceDetection[0].toFixed(4)})` : `Face Not Found (${faceDetection[0].toFixed(4)})`;

		videoFrameTensor.dispose();
		resizedFrameTensor.dispose();
		normalizedFrameTensor.dispose();
		predictions.forEach(p => p.dispose());

		this.context.clearRect(0, 0, 640, 480);
		this.context.fillColor = "#ff0000";
		for(let i = 0; i < mesh.length; i += 3){
			this.context.fillRect(mesh[i], mesh[i+1], 1, 1);
		}

		requestAnimationFrame(this.runPredictLoop);
	}
	cacheDom() {
		this.dom = {
			video: this.shadowRoot.querySelector("#video"),
			output: this.shadowRoot.querySelector("#output"),
			canvas: this.shadowRoot.querySelector("#canvas"),
			start: this.shadowRoot.querySelector("#start")
		};
	}
	attachEvents() {
		this.dom.start.addEventListener("click", this.startCamera);
		this.shadowRoot.querySelector("#predict").addEventListener("click", () => this.runPredictLoop.call(this));
	}
	attributeChangedCallback(name, oldValue, newValue) {
		this[name] = newValue;
	}
}

customElements.define("face-example", FaceExample);
