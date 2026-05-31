import { useState } from 'react'
import axios from 'axios'
import './index.css'

function App() {
  const [formData, setFormData] = useState({
    LotArea: '',
    BedroomAbvGr: '',
    FullBath: '',
    HouseStyle: '1Story',
    Age: ''
  })
  
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Send a POST request to our FastAPI backend
      const response = await axios.post('http://127.0.0.1:8000/predict', {
        LotArea: parseInt(formData.LotArea),
        BedroomAbvGr: parseInt(formData.BedroomAbvGr),
        FullBath: parseInt(formData.FullBath),
        HouseStyle: formData.HouseStyle,
        Age: parseInt(formData.Age)
      })

      setResult(response.data.predicted_price)
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        // FastAPI validation errors are usually in an array under err.response.data.detail
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) {
          setError(detail[0].msg);
        } else {
          setError(detail);
        }
      } else {
        setError("Failed to connect to the prediction server. Is FastAPI running?")
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="blob blob-1"></div>
      <div className="blob blob-2"></div>
      
      <div className="app-container">
        <div className="header-text">
          <h1>Price Predictor</h1>
          <p>AI-Powered Real Estate Valuation</p>
        </div>

        <div className="glass-card">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Lot Area (sq ft)</label>
              <input 
                type="number" 
                name="LotArea" 
                value={formData.LotArea} 
                onChange={handleChange} 
                placeholder="e.g. 8500"
                min="100"
                required 
              />
            </div>

            <div className="form-group">
              <label>Bedrooms</label>
              <input 
                type="number" 
                name="BedroomAbvGr" 
                value={formData.BedroomAbvGr} 
                onChange={handleChange} 
                placeholder="e.g. 3"
                min="0"
                max="20"
                required 
              />
            </div>

            <div className="form-group">
              <label>Bathrooms</label>
              <input 
                type="number" 
                name="FullBath" 
                value={formData.FullBath} 
                onChange={handleChange} 
                placeholder="e.g. 2"
                min="0"
                max="10"
                required 
              />
            </div>

            <div className="form-group">
              <label>House Style</label>
              <select name="HouseStyle" value={formData.HouseStyle} onChange={handleChange}>
                <option value="1Story">1 Story</option>
                <option value="1.5Fin">1.5 Story (Finished)</option>
                <option value="1.5Unf">1.5 Story (Unfinished)</option>
                <option value="2Story">2 Story</option>
                <option value="2.5Fin">2.5 Story (Finished)</option>
                <option value="2.5Unf">2.5 Story (Unfinished)</option>
                <option value="SFoyer">Split Foyer</option>
                <option value="SLvl">Split Level</option>
              </select>
            </div>

            <div className="form-group">
              <label>Age (Years)</label>
              <input 
                type="number" 
                name="Age" 
                value={formData.Age} 
                onChange={handleChange} 
                placeholder="e.g. 15"
                min="0"
                max="200"
                required 
              />
            </div>

            <button type="submit" className="btn-predict" disabled={loading}>
              {loading ? <span className="spinner"></span> : 'Predict Price'}
            </button>
          </form>

          {error && (
            <div className="error-box">
              ⚠️ {error}
            </div>
          )}

          {result && !error && (
            <div className="result-box">
              <div className="result-label">Estimated Value</div>
              <div className="result-price">
                ${result.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default App
