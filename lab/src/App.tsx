import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import PaperLabs from './pages/PaperLabs'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/papers" element={<PaperLabs />} />
      </Routes>
    </Layout>
  )
}
