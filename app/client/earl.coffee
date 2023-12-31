#########
# Earl
# The MIT License (MIT)
# Copyright (c) Travis Kriplean, Consider.it LLC
# 
# Hire Earl to handle the browser history & location bar. 
#
# When your web application is loaded, Earl will update your application's 
# retrieve('location') state with the url, params, and anchor. Your application 
# can simply react to location state changes as it would any other state change. 
#
# If your application changes the location state, Earl will dutifully update 
# the browser history/location to reflect this change. 
#
# Earl also likes to brag about the souped-up, history-aware dom.A link he 
# offers. It is a drop in replacement for dom.A that will cause internal links
# on your site to update state and browser location without loading a new page.
# Make sure to tell him you want it by setting a history-aware-link attribute
# on the script tag you use to request Earl to attend to your page:
#
# <script src="/path/to/earl.js" history-aware-links></script>
#
# If the root of your app is served off of a directory, let Earl know by adding 
# a script attribute, like: 
#
# <script src="/path/to/earl.js" root="path/served/at"></script>
#
# DISCLAIMER: Earl assumes his clients are html5 pushstate history compatible. 
# If you want to serve older non-pushstate compatible browsers try installing the 
# https://github.com/devote/HTML5-History-API polyfill first. 
#
#
# Aside from the public API, you can communicate with Earl through this state:
#    retrieve('location')
#         url: the current browser location
#         query_params: browser search values (e.g. blah?foo=fab&bar=nice)
#         hash: the anchor tag (if any) in the link. e.g. blah.html#hash
#         title: the window title


###################################
# Earl's API (Admissible Professional Inquiries): 
# 
#   Earl.load_page
#     Convenience method for changing the page's url


window.get_script_attr ?= (script, attr) ->
  sc = document.querySelector("script[src*='#{script}'][src$='.coffee'], script[src*='#{script}'][src$='.js']")
  if !sc 
    return false
  val = sc?.getAttribute(attr)
  if val == ''
    val = true 
  val 


hist_aware = if window.history_aware_links?
               window.history_aware_links
             else  
               !!get_script_attr('earl', 'history-aware-links')  

onload = -> 

  Earl.root = get_script_attr('earl', 'root') or '/'
  if Earl.root == true # earl attribute just set with no value
    Earl.root = '/'

  if window.location.pathname.match('.html')
    Earl.root += location.pathname.match(/\/([\w-_]+\.html)/)[1] + '/'
  if Earl.root[0] != '/'
    Earl.root = '/' + Earl.root

  # Earl, don't forget to update us if the browser back or forward button pressed
  window.addEventListener 'popstate', (ev) -> 
    Earl.load_page url_from_browser_location()

  # By all means Earl, please do initialize location state
  Earl.load_page url_from_browser_location()

  # Earl, don't fall asleep on the job!
  react_to_location()

window.addEventListener?('load', onload, false) or window.attachEvent?('onload', onload)



window.Earl =
  seek_to_hash: false 


  root: '/'

  # Updating the browser window location. 
  load_page: (url, query_params) ->
    loc = retrieve('location')
    loc.host = window.location.host
    loc.query_params = query_params or {}

    parser = document.createElement('a');
    parser.href = url
    
    # if the url has query parameters, parse and merge them into params
    
    if parser.search
      query_params = parser.search.replace('?', '')
      for query_param in query_params.split('&')
        query_param = query_param.split('=')
        if query_param.length == 2
          loc.query_params[query_param[0]] = query_param[1]

    # ...and parse anchors
    hash = parser.hash.replace('#', '')
    # When loading a page with a hash, we need to scroll the page
    # to proper element represented by that id. This is hard to 
    # represent in Statebus, as it is more of an event than state.
    # We'll set seek_to_hash here, then it will get set to null 
    # after it is processed. 
    Earl.seek_to_hash = hash?.length > 0

    url = parser.pathname
    url = '/' if !url || url == ''
    path = url 
    if path.substring(0, Earl.root.length) != Earl.root
      path = Earl.root + '/' + path 

    loc.path = path.replace('//', '/')
    loc.url = path.substring(Earl.root.length).replace('//', '/')
    loc.hash = hash


    if save? 
      save loc
    else 
      wait_for_load = ->
        if save? 
          save loc 
        else 
          setTimeout wait_for_load, 25
      wait_for_load()



##################################
# Internal


# Enables history aware link. Wraps basic dom.A.

if hist_aware
  window.dom = window.dom || {}

  dom.A = ->
    props = @props
    loc = retrieve 'location'
    
    if @props.href && !@props.stay_away_Earl
      href = @props.href

      # Earl will call a click handler that the programmer passes in
      onClick = @props.onClick or (-> null)      

      # Earl rewrites the address when necessary 
      internal_link = !href.match('//') || !!href.match(location.origin)
      is_mailto = !!href.toLowerCase().match('mailto')
      if internal_link && !is_mailto
        if href.substring(0, Earl.root.length) != Earl.root
          rel = href[0] != '/'
          if rel 
            href = loc.path + '/' + href 
          else  
            href = Earl.root + href 
          href = @props.href = href.replace('//', '/')


      handle_click = (event) =>
        opened_in_new_tab = event.altKey  || \
                            event.ctrlKey || \
                            event.metaKey || \
                            event.shiftKey


        # In his wisdom, Earl sometimes just lets the default behavior occur
        if !internal_link || opened_in_new_tab || is_mailto || @props.target == '_blank'
          onClick event

        # ... but other times Earl is history aware
        else 
          event.preventDefault()
          event.stopPropagation()

          Earl.load_page href
          onClick event

          # Upon navigation to a new page, it is conventional to be scrolled
          # to the top of that page. Earl obliges. Pass noScroll:true to 
          # the history aware dom.A if you don't want Earl to do this. 
          window.scrollTo(0, 0) if !@props.noScroll            
                          
          return false

      if is_mobile
        @props.onTouchEnd = (e) -> 
          # Earl won't make you follow the link if you're in the middle of swipping
          if !Earl._user_swipping
            handle_click e

        if is_android_browser # Earl's least favorite browser to support...
          @props.onClick = (e) -> e.preventDefault(); e.stopPropagation()

      else
        @props.onClick = handle_click

    React.DOM.a props, props.children


# Earl's Reactive nerves keep him vigilant in making sure that changes in location
# state are reflected in the browser history. Earl also updates the window title 
# for you, free of charge, if you set retrieve('location').title.

react_to_location = -> 
  monitor = bus.reactive -> 

    loc = retrieve 'location'

    # Update the window title if it has changed
    title = location.title or document.title
    if title && title != location.title
      document.title = title

    # Respond to a location change
    new_location = url_from_statebus()

    if @last_location != new_location 

      # update browser history if it hasn't already been updated
      if url_from_browser_location() != new_location
        l = new_location.replace(/(\/){2,}/, '/').replace(/(\/)$/, '')
        l = '/' if l == ''
        history.pushState loc.query_params, title, l

      @last_location = new_location

    # If someone clicked a link with an anchor, Earl strives to scroll
    # the page to that element. Unfortunately, Earl isn't powerful 
    # enough to deal with the mightly Webkit browser's imposition of 
    # a remembered scroll position for a return visitor upon initial 
    # page load!
    if Earl.seek_to_hash && !@int

      @int = setInterval -> 
        Earl.seek_to_hash = false
        el = document.getElementById("#{loc.hash}") or document.querySelector("[name='#{loc.hash}']")
        if el
          window.scrollTo 0, getCoords(el).top - 50

          if loc.query_params.c 
            delete loc.query_params.c 
            delete loc.hash
            save loc

          clearInterval @int 
          @int = null

      , 50

  monitor()


url_from_browser_location = -> 
  # location.search returns the query parameters

  # fix url encoding
  search = location.search?.replace(/\%2[fF]/g, '/')
  loc = location.pathname?.replace(/\%20/g, ' ')

  "#{loc}#{search}#{location.hash}"

url_from_statebus = ->
  loc = retrieve 'location'

  url = loc.path or '/'
  if loc.query_params && Object.keys(loc.query_params).length > 0
    query_params = ("#{k}=#{v}" for own k,v of loc.query_params)
    url += "?#{query_params.join('&')}" 
  if loc.hash?.length > 0
    url += "##{loc.hash}"

  url


# For handling device-specific annoyances
window.addEventListener 'ontouchstart', (e) -> Earl._user_swipping = true
window.addEventListener 'ontouchend',   (e) -> Earl._user_swipping = false
rxaosp = window.navigator.userAgent.match /Android.*AppleWebKit\/([\d.]+)/ 
is_android_browser = !!(rxaosp && rxaosp[1]<537)
ua = navigator.userAgent
is_mobile = is_android_browser || \
  ua.match(/(Android|webOS|iPhone|iPad|iPod|BlackBerry|Windows Phone)/i)
