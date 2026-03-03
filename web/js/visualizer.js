/**
 * Three.js 3D Visualizer for BiLSTM Neural Architecture
 */

class NeuralVisualizer {
    constructor() {
        this.container = document.getElementById('three-canvas');
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        
        this.nodes = [];
        this.links = [];
        this.particles = [];
        
        this.init();
        this.createArchitecture();
        this.animate();
        
        window.addEventListener('resize', () => this.onWindowResize());
    }

    init() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        this.camera.position.z = 15;
        this.camera.position.y = 2;
        
        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        this.scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0x6366f1, 5, 20);
        pointLight.position.set(5, 5, 5);
        this.scene.add(pointLight);
    }

    createArchitecture() {
        // Layers configuration
        const layers = [
            { id: 'input', count: 5, color: 0x3b82f6, x: -8 },
            { id: 'forward', count: 8, color: 0x6366f1, x: -3 },
            { id: 'backward', count: 8, color: 0xa855f7, x: 2 },
            { id: 'output', count: 3, color: 0xf43f5e, x: 7 }
        ];

        const nodeGeometry = new THREE.SphereGeometry(0.3, 32, 32);

        layers.forEach((layer, layerIdx) => {
            const layerNodes = [];
            for (let i = 0; i < layer.count; i++) {
                const material = new THREE.MeshPhongMaterial({
                    color: layer.color,
                    emissive: layer.color,
                    emissiveIntensity: 0.5,
                    shininess: 100
                });
                const node = new THREE.Mesh(nodeGeometry, material);
                
                // Position nodes vertically centered
                const yPos = (i - (layer.count - 1) / 2) * 1.5;
                node.position.set(layer.x, yPos, 0);
                
                this.scene.add(node);
                layerNodes.push(node);
                this.nodes.push(node);
            }
            
            // Link to next layer
            if (layerIdx < layers.length - 1) {
                const nextLayer = layers[layerIdx + 1];
                const nextX = nextLayer.x;
                const nextCount = nextLayer.count;
                
                layerNodes.forEach((node, nodeIdx) => {
                    // Create some random connections to visualize the "dense" or "lstm" flow
                    for (let j = 0; j < nextCount; j++) {
                        if (Math.random() > 0.6) {
                            const yNext = (j - (nextCount - 1) / 2) * 1.5;
                            this.createLink(node.position, new THREE.Vector3(nextX, yNext, 0), layer.color);
                        }
                    }
                });
            }
        });
        
        // Add some background particles
        this.createBackgroundParticles();
    }

    createLink(start, end, color) {
        const material = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.2
        });
        
        const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
        const line = new THREE.Line(geometry, material);
        this.scene.add(line);
        this.links.push({ line, start, end, color });
    }

    createBackgroundParticles() {
        const geo = new THREE.BufferGeometry();
        const counts = 1000;
        const positions = new Float32Array(counts * 3);
        
        for (let i = 0; i < counts * 3; i++) {
            positions[i] = (Math.random() - 0.5) * 50;
        }
        
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.PointsMaterial({ color: 0x444444, size: 0.1 });
        const points = new THREE.Points(geo, mat);
        this.scene.add(points);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Pulse effect
        const time = Date.now() * 0.002;
        this.nodes.forEach((node, i) => {
            const scale = 1 + Math.sin(time + i) * 0.1;
            node.scale.set(scale, scale, scale);
        });
        
        // Link interaction
        this.links.forEach((link, i) => {
            link.line.material.opacity = 0.1 + Math.sin(time * 0.5 + i) * 0.05;
        });

        // Rotate scene slightly for depth
        this.scene.rotation.y = Math.sin(time * 0.1) * 0.1;
        this.scene.rotation.x = Math.cos(time * 0.1) * 0.05;

        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Start visualizer
new NeuralVisualizer();
