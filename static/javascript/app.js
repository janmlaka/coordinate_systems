let scene;
let camera;
let renderer;
let zoomSpeed = 0.5; // You can adjust the zoom speed as needed

function handleMouseWheel(event) {
  // cross-browser wheel delta
  const delta = Math.max(-1, Math.min(1, (event.wheelDelta || -event.detail)));

  // Zoom in or out based on the delta value
  if (delta > 0) {
    camera.position.z -= zoomSpeed;
  } else {
    camera.position.z += zoomSpeed;
  }
}

function addMouseWheelListener() {
  if (document.addEventListener) {
    // For modern browsers
    document.addEventListener("mousewheel", handleMouseWheel, false);
    document.addEventListener("DOMMouseScroll", handleMouseWheel, false);
  } else {
    // For old IE
    document.attachEvent("onmousewheel", handleMouseWheel);
  }
}



function main()
  {
    addMouseWheelListener();
    const earthTexture = new THREE.TextureLoader().load('static/images/earth.png');

    const canvas = document.querySelector('#c');


    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 950);
    camera.position.z = 2;
    scene.add(camera);

    renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true,});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    renderer.autoClear = false;
    renderer.setClearColor(0x00000, 0.0);


    // create earthgeometry

    const earthgeometry = new THREE.SphereGeometry(0.6,20,20);

    const eatrhmaterial = new THREE.MeshPhongMaterial({
        roughness : 1,
        metalness:0,
        map: THREE.ImageUtils.loadTexture('static/images/earthmap1k.jpg'),
        bumpMap: THREE.ImageUtils.loadTexture('static/images/earthbump.jpg'),
        bumpScale: 0.3,
    });

    const earthmesh = new THREE.Mesh(earthgeometry,eatrhmaterial);

    scene.add(earthmesh);

    // set ambientlight

    const ambientlight = new THREE.AmbientLight(0xffffff, 1.5);
    scene.add(ambientlight);

    // set point light

    const pointerlight =  new THREE.PointLight(0xffffff,0.9);

    // set light position

    pointerlight.position.set(5,3,5);
    scene.add(pointerlight);

    // cloud
    const cloudgeometry =  new THREE.SphereGeometry(0.63,32,32);

    const cloudmaterial = new THREE.MeshPhongMaterial({
        map: THREE.ImageUtils.loadTexture('static/images/earthCloud.png'),
        transparent: true
    });

    const cloudmesh = new THREE.Mesh(cloudgeometry,cloudmaterial);

    scene.add(cloudmesh);


    // star

    const stargeometry =  new THREE.SphereGeometry(80,64,64);

    const starmaterial = new THREE.MeshBasicMaterial({

        map: THREE.ImageUtils.loadTexture('static/images/galaxy.png'),
        side: THREE.BackSide
    });

    const starmesh = new THREE.Mesh(stargeometry,starmaterial);

    scene.add(starmesh);

    const animate = () =>{
        requestAnimationFrame(animate);
        earthmesh.rotation.y -= 0.0015;
        cloudmesh.rotation.y += 0.0015;
        starmesh.rotation.y += 0.0005;

        render();
    }

    const render = () => {
        renderer.render(scene,camera);
    }

    animate();
}

window.onload = main;