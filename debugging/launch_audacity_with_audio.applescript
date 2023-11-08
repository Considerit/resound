
on importFile(fileToImport, workingDirectory, firstTime)
    tell application "System Events"
        keystroke "i" using {shift down, command down}
        delay 1

        if firstTime then
            -- Assuming the dialog is now open...
            keystroke "g" using {shift down, command down} -- Opens the 'Go to the folder:' dialog
            delay 1 -- Wait for the dialog to open
            keystroke workingDirectory -- Type the path to the directory
            delay 1 -- Wait for the path to be typed
            keystroke return -- Press return to go to the directory
            delay 1 -- Wait for the navigation
        end if

        log fileToImport

        -- Now you need to select the file
        -- This is highly dependent on how the files are listed in the dialog
        -- Below is a very generic approach that might not work without adjustments
        keystroke fileToImport -- Type the file name
        delay .5 -- Wait for the file name to be typed
        keystroke return -- Press return to select the file and close the dialog
        delay .5

    end tell
end importFile

on run (inputParameters)
    if inputParameters is missing value then set inputParameters to {}

    -- Define default values
    set defaultDirectory to ""
    set defaultFileList to {""}

    -- Set workingDirectory and fileList with input parameters or use defaults
    if (count of inputParameters) >= 1 then
        set workingDirectory to item 1 of inputParameters
    else
        set workingDirectory to defaultDirectory
    end if
    
    if (count of inputParameters) ≥ 2 then
        set fileListString to item 2 of inputParameters
        -- Convert the comma-separated string into a list
        set AppleScript's text item delimiters to ", "
        set fileList to every text item of fileListString
        set cleanedFileList to {}
        repeat with fname in fileList
            set fname to contents of fname -- dereferences the list item, turning it from {item} to item
            if fname starts with "\"" then set fname to text 2 thru -1 of fname
            if fname ends with "\"" then set fname to text 1 thru -2 of fname
            copy fname to the end of cleanedFileList
        end repeat
        set fileList to cleanedFileList


    else
        set fileList to defaultFileList
    end if


    tell application "Audacity"
        activate
    end tell

    tell application "System Events"
        keystroke "n" using {command down}
        delay 1
    end tell


    tell application "System Events"
        set firstTime to true
        repeat with aFile in fileList

            -- Check if Audacity is still the active application
            set frontApp to name of first application process whose frontmost is true
            if frontApp is not "Audacity" then
                -- If Audacity is not the frontmost application, exit the loop
                exit repeat
            end if

            my importFile(aFile, workingDirectory, firstTime)
            set firstTime to false
        end repeat

    end tell
end run