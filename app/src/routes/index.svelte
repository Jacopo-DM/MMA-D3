<script lang="ts">
	console.clear();
	import * as THREE from 'three';
	import * as SC from 'svelte-cubed';

	import { slate_50, slate_900 } from '../assets/palette.json';

	import Mesh from './Mesh.svelte';

	// Init
	// const fractionToUse = 0.1;
	// const numOfPaintings = Math.floor(23246 * fractionToUse);
	// const numOfSides = 6;
	// const numOfMeshes = Math.floor(numOfPaintings / numOfSides);

	// Init
	const spacing = 1.2;
	// const factor = Math.floor(Math.sqrt(numOfPaintings));

	// const center = Math.floor((factor / 2) * spacing);
	// const distCam = Math.floor(Math.sqrt(numOfPaintings));
	const center = 5 * spacing;
	const distCam = 5;
	const xc = center;
	const yc = center;
</script>

<SC.Canvas antialias={true} background={new THREE.Color(slate_900)}>
	<!-- Insert -->
	<Mesh imageFolder={'smallest'} imageExt={'jpg'} />

	<!-- Scene -->
	<SC.Group>
		<SC.Primitive object={new THREE.AmbientLight(slate_50, 0.65)} />
		<SC.Primitive object={new THREE.DirectionalLight(slate_50, 0.65)} position={[0, 10, 10]} />
	</SC.Group>

	<SC.PerspectiveCamera position={[xc, distCam, yc]} target={[xc, 0, yc]} fov={90} near={0.1} />
	<SC.OrbitControls
		target={[xc, 0, yc]}
		enabled={true}
		enableRotate={false}
		enableDamping={true}
		dampingFactor={0.15}
		mouseButtons={{
			LEFT: THREE.MOUSE.PAN,
			MIDDLE: THREE.MOUSE.DOLLY,
			RIGHT: THREE.MOUSE.ROTATE
		}}
	/>
</SC.Canvas>
