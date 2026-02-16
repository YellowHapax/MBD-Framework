import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import CubeLab from './pages/CubeLab'
import BaselineLab from './pages/BaselineLab'
import FieldLab from './pages/FieldLab'
import CouplingLab from './pages/CouplingLab'
import SocialLab from './pages/SocialLab'
import AgentLab from './pages/AgentLab'
import ResonanceLab from './pages/ResonanceLab'
import PaperLabs from './pages/PaperLabs'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/papers" element={<PaperLabs />} />
        <Route path="/cube" element={<CubeLab />} />
        <Route path="/baseline" element={<BaselineLab />} />
        <Route path="/fields" element={<FieldLab />} />
        <Route path="/coupling" element={<CouplingLab />} />
        <Route path="/social" element={<SocialLab />} />
        <Route path="/agent" element={<AgentLab />} />
        <Route path="/resonance" element={<ResonanceLab />} />
      </Routes>
    </Layout>
  )
}
