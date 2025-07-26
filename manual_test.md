# Manual Testing Checklist for HIIT Analyzer

## Prerequisites
- [ ] Application is running on http://localhost:8050
- [ ] Browser developer console is open (F12)

## Test 1: Initial Page Load
- [ ] Navigate to http://localhost:8050
- [ ] Page loads without errors
- [ ] Title says "HIIT Analyzer"
- [ ] Navigation bar is visible with tabs: Raw Data, Interval Analysis, Performance Metrics, Plotly JS Demo
- [ ] No JavaScript errors in console

## Test 2: Raw Data Tab
- [ ] Click on "Raw Data" tab
- [ ] Page loads without errors
- [ ] File dropdown selector is visible
- [ ] If no .fit files: Shows "No .fit files found" message
- [ ] If .fit files exist:
  - [ ] Files appear in dropdown
  - [ ] Selecting a file loads data
  - [ ] Summary statistics table appears
  - [ ] Time series plots appear
  - [ ] No JavaScript errors in console

## Test 3: Interval Analysis Tab  
- [ ] Click on "Interval Analysis" tab
- [ ] Page loads without errors
- [ ] File dropdown selector is visible
- [ ] If a file is selected:
  - [ ] HIIT detection plot appears
  - [ ] Frequency analysis section appears
  - [ ] Interval overlay plot appears
  - [ ] Metrics table appears
  - [ ] No JavaScript errors in console

## Test 4: Performance Metrics Tab
- [ ] Click on "Performance Metrics" tab
- [ ] Page loads without errors
- [ ] If multiple files exist:
  - [ ] Performance comparison plots appear
  - [ ] Summary statistics appear
- [ ] No JavaScript errors in console

## Test 5: Interactive Features
- [ ] In Interval Analysis, threshold slider works
- [ ] Save/Clear threshold buttons work
- [ ] File selection persists when switching tabs

## Common Issues to Check
1. **JSON Serialization Error**: If you see "TypeError: Object of type int64 is not JSON serializable"
   - This means NumPy types aren't being converted properly
   
2. **Import Errors**: If modules fail to import
   - Check that all dependencies are installed
   - Check for circular imports

3. **No Data Display**: If pages load but show no data
   - Check that .fit files are in the data/ directory
   - Check browser console for errors

## Expected Behavior
- Application should load all pages without errors
- File selection should work across all tabs
- All visualizations should render properly
- No console errors should appear