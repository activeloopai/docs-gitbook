---
description: How to embed our visualizer in your application.
---

# Visualizer Integration

## How to embed the Activeloop visualizer into your own web applications

Visualization engine allows the user to visualize, explore, and interact with Deep Lake datasets. In addition to using through the [Activeloop UI](https://app.activeloop.ai/) or [in Python](../getting-started/deep-learning/visualizing-datasets.md), the Activeloop visualizer can also be embedded into your application.

### HTML iframe (Alpha)

To embed into your html page, you can use our iframe integration:

```html
<iframe src="https://app.activeloop.ai/visualizer/iframe?url=hub://activeloop/imagenet-train" width="800px" height="600px">
```

**iframe URL:** `https://app.activeloop.ai/visualizer/iframe?url=hub://$org/$ds&{checkpoint=$checkpoint}&{vs=$visualizer_state}&{token=$token}`

**Params:**

`url` - The url of the dataset \
`vs` - Visualizer state, which can be obtained from the platform url \
`token` - User token, for private datasets. If the value is \
`ask` then the UI will be populated for entering the token \
`checkpoint` - Dataset checkpoint \
`query` - Query string to apply on the dataset

### Javascript API (Alpha)

To have more fine grained control, you can embed the visualizer using Javascript:

```html
<div id='container'></div>
<script src="https://app.activeloop.ai/visualizer/vis.js"></script>
<script>
  let container = document.getElementById('container')
  window.vis.visualize("hub://activeloop/imagenet-train", null, null, container, null)
</script>
```

or to visualize private datasets with authentication

```html
<div id='container'></div>
<script src="https://app.activeloop.ai/visualizer/vis.js"></script>
<script>
  let container = document.getElementById('container')
  window.vis.visualize("hub://org/private", null, null, container, {
		requireSignin: true
	})
</script>
```

**Interface**

Below you can find definitions of the arguments.

```javascript
/// ds - Dataset url
/// commit - optional commit id
/// state - optional initial state of the visualizer
/// container - HTML element serving as container for visualizer elements
/// options - optional Visualization options
static visualize(
  ds: string,
  commit: string | null = null,
  state: string | null = null,
  container: HTMLElement,
  options: VisOptions | null
): Promise<Vis>;

/// backlink - Show backlink to platform button
/// singleSampleView - Enable single sample view through enter key
/// requireSignin - Requires signin to get access token
/// token - Token id
/// gridMode - Canvas vs Grid
/// queryString - Query to apply on the iframe
export type VisOptions = {
  backlink?: Boolean
  singleSampleView?: Boolean
  requireSignin?: Boolean
  token: string | null
  gridMode?: "canvas" | "grid"
  queryString?: string
}
```

This `visualize` returns `Promise<Vis>` which can be used to dynamically change the visualizer state. Vis supports only `query` functions for now

```jsx
class Vis
{
	/// Asynchronously runs a query and resolves the promise when query completed.
  /// In case of error in query, rejects the promise.
	query(queryString: string): Promise<void>
}
```



