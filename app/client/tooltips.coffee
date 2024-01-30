
window.$$ = 

  add_delegated_listener: (el, eventName, selector, callback) ->

    el.addEventListener eventName, (e) ->
      try 
        target = e.target
        while target && target != el
          if target.matches selector
            callback.call target, e
            break
          target = target.parentNode
      catch e
        console.log('Got error in add_delegated_listener')
        console.error(e)

    , false

  closest: (el, selector) -> 
    if el.closest?
      el.closest(selector)
    else 
      matchesSelector = el.matches || el.webkitMatchesSelector || el.mozMatchesSelector || el.msMatchesSelector;
      current_el = el
      while (current_el)
        if matchesSelector.call(current_el, selector)
          return current_el
        else
          current_el = current_el.parentElement

      null
  offset: (el) ->
    rect = el.getBoundingClientRect()

    top  = rect.top  + window.pageYOffset - document.documentElement.clientTop
    left = rect.left + window.pageXOffset - document.documentElement.clientLeft

    {top, left}


window.calc_coords_for_tooltip_or_popover = (el) ->
  coords = $$.offset(el)
  coords.width = el.offsetWidth
  coords.height = el.offsetHeight
  coords.left += el.offsetWidth / 2
  coords



TOOLTIP_DELAY = 500
window.clear_tooltip = ->
  tooltip = retrieve('tooltip')
  tooltip.coords = tooltip.tip = tooltip.top = tooltip.positioned = null
  tooltip.offsetY = tooltip.offsetX = null 
  tooltip.rendered_size = false 
  save tooltip

toggle_tooltip = (e) ->
  tooltip_el = $$.closest(e.target, '[data-tooltip]')
  if tooltip_el?
    tooltip = retrieve('tooltip')
    if tooltip.coords
      clear_tooltip()
    else 
      show_tooltip(e)

show_tooltip = (e) ->

  tooltip_el = $$.closest(e.target, '[data-tooltip]')
  return if !tooltip_el? || tooltip_el.style.opacity == 0

  tooltip = retrieve 'tooltip'
  name = tooltip_el.getAttribute('data-tooltip')
  if tooltip.tip != name 
    tooltip.tip = name

    setTimeout ->
      if tooltip.tip == name 
        tooltip.coords = calc_coords_for_tooltip_or_popover(tooltip_el)
        save tooltip
    , TOOLTIP_DELAY

    e.preventDefault()
    e.stopPropagation()

tooltip = retrieve 'tooltip'
hide_tooltip = (e) ->
  if e.target.getAttribute('data-tooltip')
    clear_tooltip()
    e.preventDefault()
    e.stopPropagation()


setTimeout ->
  document.addEventListener "click", toggle_tooltip

  document.body.addEventListener "mouseover", show_tooltip, true
  document.body.addEventListener "mouseleave", hide_tooltip, true

  $$.add_delegated_listener document.body, 'focusin', '[data-tooltip]', show_tooltip
  $$.add_delegated_listener document.body, 'focusout', '[data-tooltip]', hide_tooltip
, 1000



window.get_tooltip_or_popover_position = ({popover, tooltip, arrow_size})->
  popover ?= tooltip

  coords = popover.coords
  tip = popover.tip

  window_height = window.innerHeight
  viewport_top = window.scrollY
  viewport_bottom = viewport_top + window_height


  arrow_up = popover.top || !popover.top?
  try_top = (force) -> 

    top = coords.top + (popover.offsetY or 0) - (popover.rendered_size?.height or 0) - arrow_size.height - 6

    if top < viewport_top && !force
      arrow_up = false
      return null
    else 
      arrow_up = true
      top

  try_bottom = (force) -> 
    top = coords.top + (popover.offsetY or 0) + arrow_size.height + coords.height + 12
    if top + (popover.rendered_size?.height or 0) + arrow_size.height > viewport_bottom && !force
      arrow_up = true
      return null
    else
      arrow_up = false 
      top

  if popover.top || !popover.top? 
    # place the popover above the element
    top = try_top()
    if top == null 
      top = try_bottom()
      if top == null
        top = try_top(true)
  else 
    # place the popover below the element
    top = try_bottom()
    if top == null 
      top = try_top()
      if top == null
        top = try_bottom(true)


  arrow_adjustment = 0 
  if popover.rendered_size?.width
    left = coords.left + (popover.offsetX or 0) - popover.rendered_size.width / 2

    if left < 0
      arrow_adjustment = -1 * left
      left = 0
    else if left + popover.rendered_size.width > window.innerWidth
      arrow_adjustment = (window.innerWidth - popover.rendered_size.width) - left
      left = window.innerWidth - popover.rendered_size.width

  else 
    left = -999999 # render it offscreen first to get sizing


  {top, left, arrow_up, arrow_adjustment}






console.log("HI!")
dom.TOOLTIP = ->


  tooltip = retrieve('tooltip')
  return SPAN(null) if !tooltip.coords

  coords = tooltip.coords
  tip = tooltip.tip

  arrow_size = 
    height: 7
    width: 14

  {top, left, arrow_up, arrow_adjustment} = get_tooltip_or_popover_position({tooltip, arrow_size})

  style = defaults {top, left}, (@props.style or {}), 
    fontSize: 14
    padding: '4px 8px'
    borderRadius: 8
    pointerEvents: 'none'
    zIndex: 999999999999
    color: 'white'
    backgroundColor: 'black'
    position: 'absolute'      
    boxShadow: '0 1px 1px rgba(0,0,0,.2)'
    maxWidth: 350

  DIV
    id: 'tooltip'
    role: "tooltip"
    style: style


    DIV 
      dangerouslySetInnerHTML: {__html: tip}


    SVG 
      width: arrow_size.width
      height: arrow_size.height 
      viewBox: "0 0 531.74 460.5"
      preserveAspectRatio: "none"
      style: 
        position: 'absolute'
        bottom: if arrow_up then -arrow_size.height
        top: if !arrow_up then -arrow_size.height
        left: if tooltip.positioned != 'right' then "calc(50% - #{arrow_size.width / 2 + arrow_adjustment}px)" 
        right: if tooltip.positioned == 'right' then 7       
        transform: if !arrow_up then 'scale(1,-1)' 
        display: if tooltip.hide_triangle then 'none' 

      POLYGON
        stroke: "black" 
        fill: 'black'
        points: "530.874,0.5 265.87,459.5 0.866,0.5"


dom.TOOLTIP.refresh = ->
  tooltip = retrieve('tooltip')
  if !tooltip.rendered_size && tooltip.coords 

    tooltip.rendered_size = 
      width: @getDOMNode().offsetWidth
      height: @getDOMNode().offsetHeight
    save tooltip

