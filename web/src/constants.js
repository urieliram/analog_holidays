// ForecastEnergAI brand palettes
// forecastenergai_colors.json + cohesive_colors.json

export const BRAND = {
  electric_green: '#00C853',
  deep_purple:    '#6A1B9A',
  white:          '#FFFFFF',
  dark_gray:      '#212121',
  // cohesive palette
  bright_teal:    '#06d596',
  dark_teal:      '#014754',
  violet:         '#9274ff',
}

export const UI = {
  bg:          '#ffffff',
  bg_soft:     '#f5f9fa',
  text_main:   '#2f3b40',
  text_sec:    '#5f6f75',
  border:      '#e3eef0',
  header_bg:   '#014754',   // dark_teal
  accent:      '#06d596',   // bright_teal
  analytics:   '#9274ff',   // violet
}

export const LABEL_COLOR = {
  holiday:     '#2ca02c',
  special_day: '#d62728',
  normal_day:  '#1f77b4',
}

export const LABEL_BG = {
  holiday:     '#c3e6cb',
  special_day: '#f5c6cb',
  normal_day:  '#cce5ff',
}

export const LABEL_TEXT_COLOR = {
  holiday:     '#155724',
  special_day: '#721c24',
  normal_day:  '#004085',
}

export const LABEL_SYMBOL = {
  holiday:     '🟢',
  special_day: '🔴',
  normal_day:  '🔵',
}

export const MONTH_ABBR_EN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export const DOW_SHORT_EN  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

export const REGIONS = [
  'SEN_demand_SIN',
  'SEN_demand_CEL',
  'SEN_demand_NOR',
  'SEN_demand_PEN',
  'SEN_demand_ORI',
  'SEN_demand_NES',
  'SEN_demand_NTE',
]

export const HOURS = Array.from({ length: 24 }, (_, i) => i)
