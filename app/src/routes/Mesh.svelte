<script lang="ts">
	import * as THREE from 'three';
	import * as SC from 'svelte-cubed';
	import { tweened } from 'svelte/motion';
	import { cubicOut } from 'svelte/easing';

	import pm from '../assets/painting.json';

	// Init
	const fractionToUse = 0.05;
	const numOfPaintings = Math.floor(23246 * fractionToUse);
	const numOfSides = 6;
	const numOfMeshes = Math.floor(numOfPaintings / numOfSides);

	// Init
	const spacing = 1.2;
	const factor = Math.floor(Math.sqrt(numOfPaintings));

	let IMeshes: any = [];
	// Data
	const data = new Array(numOfSides * numOfMeshes).fill(0).map((d, id) => {
		const x = (id % factor) * spacing;
		const z = Math.floor(id / factor) * spacing;
		return [x, 0, z];
	});

	// Primitives
	const texture = new THREE.TextureLoader();
	const geometry = new THREE.BoxBufferGeometry(pm.w, pm.h);

	//  Rotation correction
	let d2r = Math.PI / 180;
	var maps: any = {
		0: [0, -90 * d2r, 90 * d2r, 0, 1, 1],
		1: [0, 90 * d2r, 270 * d2r, 0, 1, 1],
		2: [0, 0, 0, 1, 0, 1],
		3: [0, 180 * d2r, 180 * d2r, 1, 0, 1],
		4: [90 * d2r, 180 * d2r, 180 * d2r, 1, 1, 0],
		5: [90 * d2r, 0, 180 * d2r, 1, 1, 0]
	};

	//  Load textures
	let material: any = [];
	let count = 0;
	const progress = tweened(0, {
		duration: 400,
		easing: cubicOut
	});
	export let imageFolder = 'smallest';
	export let imageExt = 'jpg';
	for (let j = 0; j < numOfMeshes; j++) {
		for (let i = 0; i < numOfSides; i++) {
			let idx = i + j * numOfSides;
			const imgUrl1 = './' + imageFolder + '/' + idx + '.' + imageExt;
			// const _texture = texture.load(imgUrl1);
			const _texture = texture.load(imgUrl1, function (texture) {
				count += 1;
				progress.set(count / numOfPaintings);
				material.push(
					new THREE.MeshBasicMaterial({
						map: texture,
						transparent: true
					})
				);
			});
			// _texture.needsUpdate = true;
			material.push(
				new THREE.MeshBasicMaterial({
					map: _texture,
					transparent: true
				})
			);
		}
	}

	let IMesh: any;
	for (let j = 0; j < numOfMeshes; j++) {
		const jdx = j * numOfSides;
		IMesh = new THREE.InstancedMesh(geometry, material.slice(jdx, jdx + numOfSides), numOfSides);
		for (let i = 0; i < IMesh.count; i++) {
			const idx = i + j * numOfSides;
			const dummy = new THREE.Object3D();
			const datum = data[idx];
			const codex = maps[i % 6];

			dummy.position.set(datum[0], datum[1], datum[2]);

			dummy.updateMatrix();
			dummy.rotateX(codex[0]);
			dummy.updateMatrix();

			dummy.updateMatrix();
			dummy.rotateY(codex[1]);
			dummy.updateMatrix();

			dummy.updateMatrix();
			dummy.rotateZ(codex[2]);
			dummy.updateMatrix();

			dummy.updateMatrix();
			dummy.scale.set(codex[3], codex[4], codex[5]);
			dummy.updateMatrix();

			IMesh.setMatrixAt(i, dummy.matrix);
		}
		IMesh.material.needsUpdate = true;
		IMeshes.push(IMesh);
	}
</script>

<progress value={$progress} />

{#each IMeshes as Mesh}
	<SC.Primitive object={Mesh} />
{/each}

<style>
	progress {
		position: absolute;
		display: block;
		width: 100%;
	}
</style>
