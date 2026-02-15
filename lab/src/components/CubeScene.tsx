/**
 * CubeScene.tsx — Interactive 3D Stella Octangula
 *
 * Renders the Influence Cube: 8 vertices (4 constructive, 4 destructive)
 * inscribed in a unit cube, with the two dual tetrahedra visible as
 * edge networks.
 */

import { useRef, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Line } from '@react-three/drei'
import * as THREE from 'three'
import type { Vertex } from '../types'

/* ---------- constants ---------- */

const CONSTRUCTIVE_COLOR = '#60a5fa' // blue-400
const DESTRUCTIVE_COLOR = '#fb7185' // rose-400
const CUBE_WIRE_COLOR = '#334155'    // slate-700
const AXIS_COLORS = ['#f97316', '#a855f7', '#14b8a6'] // orange, purple, teal

/* Map 0..1 cube coords to centered -1..1 for visual */
function cubeToScene(c: [number, number, number]): [number, number, number] {
  return [c[0] * 2 - 1, c[1] * 2 - 1, c[2] * 2 - 1]
}

/* ---------- sub-components ---------- */

function VertexSphere({
  vertex,
  position,
  selected,
  onSelect,
}: {
  vertex: Vertex
  position: [number, number, number]
  selected: boolean
  onSelect: () => void
}) {
  const ref = useRef<THREE.Mesh>(null!)
  const color = vertex.constructive ? CONSTRUCTIVE_COLOR : DESTRUCTIVE_COLOR

  useFrame(() => {
    if (ref.current) {
      const s = selected ? 1.3 : 1.0
      ref.current.scale.lerp(new THREE.Vector3(s, s, s), 0.1)
    }
  })

  return (
    <group position={position}>
      <mesh ref={ref} onClick={onSelect}>
        <sphereGeometry args={[0.08, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 0.6 : 0.2}
          roughness={0.3}
          metalness={0.1}
        />
      </mesh>
      <Text
        position={[0, 0.18, 0]}
        fontSize={0.13}
        color={color}
        anchorX="center"
        anchorY="bottom"
        font="https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfAZ9hjQ.woff2"
      >
        {vertex.name}
      </Text>
    </group>
  )
}

function TetraEdges({
  vertices,
  color,
}: {
  vertices: Vertex[]
  color: string
}) {
  const lines = useMemo(() => {
    const edges: [number, number][] = []
    for (let i = 0; i < vertices.length; i++)
      for (let j = i + 1; j < vertices.length; j++)
        edges.push([i, j])
    return edges
  }, [vertices])

  return (
    <>
      {lines.map(([i, j], idx) => {
        const a = cubeToScene(vertices[i].coords)
        const b = cubeToScene(vertices[j].coords)
        return (
          <Line
            key={idx}
            points={[a, b]}
            color={color}
            lineWidth={1.5}
            opacity={0.5}
            transparent
          />
        )
      })}
    </>
  )
}

function DualLines({ vertices }: { vertices: Vertex[] }) {
  /* Connect each vertex to its dual — dashed */
  const pairs = useMemo(() => {
    const seen = new Set<string>()
    const result: [Vertex, Vertex][] = []
    for (const v of vertices) {
      const d = vertices.find((u) => u.name === v.dual)
      if (d && !seen.has(`${v.name}-${d.name}`)) {
        seen.add(`${v.name}-${d.name}`)
        seen.add(`${d.name}-${v.name}`)
        result.push([v, d])
      }
    }
    return result
  }, [vertices])

  return (
    <>
      {pairs.map(([a, b], idx) => (
        <Line
          key={idx}
          points={[cubeToScene(a.coords), cubeToScene(b.coords)]}
          color="#8b5cf6"
          lineWidth={1}
          opacity={0.25}
          transparent
          dashed
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}
    </>
  )
}

function CubeWireframe() {
  /* Unit cube wireframe edges */
  const edges: [number, number, number][][] = []
  for (let a = -1; a <= 1; a += 2)
    for (let b = -1; b <= 1; b += 2) {
      edges.push([[a, b, -1], [a, b, 1]])
      edges.push([[a, -1, b], [a, 1, b]])
      edges.push([[-1, a, b], [1, a, b]])
    }

  return (
    <>
      {edges.map((pts, i) => (
        <Line
          key={i}
          points={pts as [number, number, number][]}
          color={CUBE_WIRE_COLOR}
          lineWidth={0.5}
          opacity={0.3}
          transparent
        />
      ))}
    </>
  )
}

function AxisLabels() {
  const labels: { text: string; pos: [number, number, number]; color: string }[] = [
    { text: 'Locus →', pos: [0, -1.35, -1.35], color: AXIS_COLORS[0] },
    { text: 'Coupling →', pos: [-1.35, 0, -1.35], color: AXIS_COLORS[1] },
    { text: 'Temporality →', pos: [-1.35, -1.35, 0], color: AXIS_COLORS[2] },
  ]

  return (
    <>
      {labels.map((l, i) => (
        <Text
          key={i}
          position={l.pos}
          fontSize={0.1}
          color={l.color}
          anchorX="center"
          font="https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfAZ9hjQ.woff2"
        >
          {l.text}
        </Text>
      ))}
    </>
  )
}

function SlowRotation({ children }: { children: React.ReactNode }) {
  const ref = useRef<THREE.Group>(null!)
  useFrame((_state, delta) => {
    ref.current.rotation.y += delta * 0.08
  })
  return <group ref={ref}>{children}</group>
}

/* ---------- main scene ---------- */

interface CubeSceneProps {
  vertices: Vertex[]
  selectedVertex: Vertex | null
  onSelectVertex: (v: Vertex | null) => void
}

export default function CubeScene({
  vertices,
  selectedVertex,
  onSelectVertex,
}: CubeSceneProps) {
  const constructive = useMemo(
    () => vertices.filter((v) => v.constructive),
    [vertices],
  )
  const destructive = useMemo(
    () => vertices.filter((v) => !v.constructive),
    [vertices],
  )

  return (
    <Canvas
      camera={{ position: [3, 2.5, 3], fov: 40 }}
      style={{ background: 'transparent' }}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.4} />
      <pointLight position={[5, 5, 5]} intensity={0.8} />
      <pointLight position={[-3, -2, 4]} intensity={0.3} color="#a78bfa" />

      <SlowRotation>
        <CubeWireframe />
        <TetraEdges vertices={constructive} color={CONSTRUCTIVE_COLOR} />
        <TetraEdges vertices={destructive} color={DESTRUCTIVE_COLOR} />
        <DualLines vertices={vertices} />
        <AxisLabels />

        {vertices.map((v) => (
          <VertexSphere
            key={v.name}
            vertex={v}
            position={cubeToScene(v.coords)}
            selected={selectedVertex?.name === v.name}
            onSelect={() =>
              onSelectVertex(selectedVertex?.name === v.name ? null : v)
            }
          />
        ))}
      </SlowRotation>

      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={3}
        maxDistance={8}
        autoRotate={false}
      />
    </Canvas>
  )
}
