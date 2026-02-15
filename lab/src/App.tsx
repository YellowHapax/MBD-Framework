import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import CubeLab from './pages/CubeLab'
import BaselineLab from './pages/BaselineLab'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/cube" element={<CubeLab />} />
        <Route path="/baseline" element={<BaselineLab />} />
      </Routes>
    </Layout>
  )
}
