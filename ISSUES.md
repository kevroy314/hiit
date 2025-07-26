# Issues Found in Functional Requirements

## 1. Directory Structure vs Existing Code
- **Issue**: Requirements specify all code should be in `src/` directory, but current codebase has code in root and `hiit/` directory
- **Resolution**: Will create new structure with all code in `src/`

## 2. Performance Analysis Page Scatter Plot Labels
- **Issue**: Requirement states dots should have labels like "1.A" where 1 is session number and A is ordinal of interval. With many sessions and intervals, this could create an unreadable plot with overlapping labels.
- **Resolution**: Will implement as specified but may need to make labels optional or only show on hover for readability

## 3. Algorithm Step Numbering
- **Issue**: Algorithm steps go from 0 to 7, but step 5 mentions using template correlation from step 4, suggesting potential numbering confusion
- **Resolution**: Will implement algorithm as written, assuming step references are correct

## 4. Functional Tests Requirement
- **Issue**: Requirement for "functional tests that can be quickly run" but no specific test framework or coverage requirements specified
- **Resolution**: Will use pytest for functional tests covering main features

## 5. URL State Management
- **Issue**: Requirement to "Include in the address bar the name of the selected file and the page the app is currently displaying" - Dash's URL routing may require specific implementation approach
- **Resolution**: Will use Dash's dcc.Location component for URL routing with query parameters

## 6. Pylint Compliance
- **Issue**: "The application should pass all pylint checks" - pylint with default settings can be very strict and may conflict with some Dash patterns
- **Resolution**: Will aim for pylint compliance but may need to disable specific warnings that conflict with Dash idioms

## 7. Magic Numbers in .env
- **Issue**: Storing numeric algorithm parameters in .env file (which treats everything as strings) requires careful type conversion
- **Resolution**: Will implement proper type conversion for all numeric parameters from .env

## 8. Cache Implementation
- **Issue**: "Results from the algorithm should be cached" but no specific caching strategy specified
- **Resolution**: Will implement file-based caching in settings directory using JSON