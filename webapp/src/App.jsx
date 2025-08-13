import { useRef, useEffect } from 'react';
import { Viewer, Entity } from 'resium';
import {
  Cartesian3,
  Color,
  GeoTIFFImageryProvider,
  Ion,
  PinBuilder,
  createWorldTerrain,
  createGooglePhotorealistic3DTileset,
  VerticalOrigin,
} from 'cesium';

Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_TOKEN || '';

const cities = [
  {
    name: 'Bologna',
    coords: [11.3426, 44.4949],
  },
  {
    name: 'London',
    coords: [-0.1276, 51.5072],
  },
];

export default function App() {
  const viewerRef = useRef();

  useEffect(() => {
    const viewer = viewerRef.current.cesiumElement;
    createGooglePhotorealistic3DTileset()
      .then((tileset) => {
        viewer.scene.primitives.add(tileset);
      })
      .catch((e) => console.error(e));
  }, []);

  const handleCityClick = (coords) => {
    const viewer = viewerRef.current.cesiumElement;
    viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(...coords, 1500),
    });
  };

  const handleFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const provider = await GeoTIFFImageryProvider.fromUrl(url, {
      projectToGeographic: true,
    });
    const viewer = viewerRef.current.cesiumElement;
    viewer.imageryLayers.addImageryProvider(provider);
  };

  const pinBuilder = new PinBuilder();

  return (
    <>
      <input
        type="file"
        accept=".tif,.tiff"
        onChange={handleFile}
        style={{ position: 'absolute', zIndex: 1, top: 10, left: 10 }}
      />
      <Viewer full ref={viewerRef} terrainProvider={createWorldTerrain()}>
        {cities.map((c) => (
          <Entity
            key={c.name}
            name={c.name}
            position={Cartesian3.fromDegrees(...c.coords)}
            onClick={() => handleCityClick(c.coords)}
            billboard={{
              image: pinBuilder.fromText(c.name[0], Color.RED, 48).toDataURL(),
              verticalOrigin: VerticalOrigin.BOTTOM,
            }}
          />
        ))}
      </Viewer>
    </>
  );
}
