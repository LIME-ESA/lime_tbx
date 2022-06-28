/*!
    @file   explorer_file_handling.h
    @brief  Public header file for the explorer_file_handling library.
    @version  4.22
    @date   22/12/2021

    <b>Copyright DEIMOS SPACE S.L.U</b>

    <b>Project: EOCFI</b>

    This product includes software developed by the GNOME Project
    http://xmlsoft.org/


*/

/*!
    @mainpage

    <center><b>Software User Manual

    EE-MA-DMS-GS-008 Issue 4.22

    22/12/2021</b></center>


    @htmlinclude usage_guide.html

*/

#ifndef _EXPLORER_FILE_HANDLING_H
#define _EXPLORER_FILE_HANDLING_H

/* Constants */
/* ========= */

/*! @def XF_MAX_FILES_NUMBER
         Maximum number of open files allowed*/
#define XF_MAX_FILES_NUMBER 10
/*! @def XF_MAX_ERROR_MSG_LENGTH
         Maximum number of characters in error message*/
#define XF_MAX_ERROR_MSG_LENGTH 512
/*! @def XF_MAX_XML_NODE_NAME_LENGTH
         Maximum number of characters in element or attribute name */
#define XF_MAX_XML_NODE_NAME_LENGTH 64
/*! @def XF_MAX_VALUE_LENGTH
         Maximum number of characters in content value of XML nodes */
#define XF_MAX_VALUE_LENGTH 512
/*! @def XF_MAX_ATTR_ARRAY_SIZE
         Maximum number of attributes */
#define XF_MAX_ATTR_ARRAY_SIZE 10
/*! @def XF_MAX_PATH_LENGTH
         Maximum number of characters in a XML path */
#define XF_MAX_PATH_LENGTH 256
/*! @def XF_MAX_FILENAME_LENGTH
         Maximum number of characters in a filename */
#define XF_MAX_FILENAME_LENGTH 64

/* Main CFI Functions - prototype declaration */
/* ========================================== */
#ifdef __cplusplus
extern "C"
{
#endif
  /* Parser initialization and clean-up functions */
  /* ============================================ */

  /*! @fn long xf_tree_init_parser ( char *file,
                                   long *error );

    @brief Loads an XML file into memory

    @param file   (IN):
             File name
    @param error (OUT):
             Error code
    @return A file descriptor of long type is returned

    This function reads an XML into memory and builds a tree representation
    of the file data ready for being accessed by the functions in the
    EO_FILE_HANDLING library. It returns a file descriptor to be used for
    referencing the file in the EO file handling routines. If the maximum
    number of open files has already been reached, an error is returned.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_MAX_FILES_REACHED
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_MEMORY_ERROR
       - XF_CFI_PARSING_FILE_ERROR
       - XF_CFI_GETTING_ROOT_ERROR
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  long xf_tree_init_parser(char* file,
                           long* error);
  /*!
    @fn void xf_tree_cleanup_parser ( long *fd, long *error );

    @brief Releases parser resources

    @param fd (IN):
             File descriptor
    @param error (OUT):
             Error code

    This function releases the memory resources taken by the XML parser for
    the corresponding file descriptor given in input.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_cleanup_parser(long* fd, long* error);

  /*!
    @fn void xf_tree_cleanup_all_parser ( void );

    @brief Releases all parser resources

    This function releases the memory resources taken by the XML parser for
    all the open files.

    This function is not thread safe and it should not be called from a 
    multi-thread program.

    - Assumptions : N/A
    - Errors   : N/A
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_cleanup_all_parser(void);

  /* Reading functions */
  /* ================= */

  /*! @fn void xf_tree_get_current_element_name ( long *fd,
                                                char **name,
                                                long *error );

    @brief Gets the name of the current element.

    @param fd (IN):
             File descriptor
    @param name   (OUT):
             Name of the current element
    @param error (OUT):
             Error code

    This function gets the name of the current element node.

    - Assumptions : 
       - It is assumed that \a *name has no memory reserved and points to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NOT_AN_ELEMENT_NODE
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for \a *name. The user must free this memory after using it.
 */
  void xf_tree_get_current_element_name(long* fd,
                                        char** name,
                                        long* error);

  /*! @fn       void xf_tree_read_integer_element_value ( long *fd,
                                                        char *element,
                                                        long *value,
                                                        long *error );
    @brief    Reads an integer number value

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name. A NULL value means an iteration.
    @param    value   (OUT):
                Element value
    @param    error   (OUT):
                Error code

    This function reads the value of the next element in the file referred by
    the \a fd whose name matches \a element and returns it as an
    integer.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NOT_A_LONG
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_read_integer_element_value(long* fd,
                                          char* element,
                                          long* value,
                                          long* error);

  /*! @fn  void xf_tree_read_integer_element_array ( long *fd,
                                                   char *element,
                                                   long **array,
                                                   long *length,
                                                   long *error );
    @brief    Reads an array of integer number values

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name
    @param    array  (OUT):
                Array of values
    @param    length (OUT):
                Actual array size
    @param    error  (OUT):
                Error code

    This function reads the values of all elements in the file referred by
    the \a fd whose name match \a element and returns them as
    an array of integers. Please noticed that in case there are several
    conversion errors in the array elements, only the last one is
    published.
    The output array is allocated internally, so the associated memory
    must be freed by the user.

    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_A_LONG
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_read_integer_element_array(long* fd,
                                          char* element,
                                          long** array, // AN-853
                                          long* length,
                                          long* error);

  /*! @fn void xf_tree_read_real_element_value ( long *fd,
                                               char *element,
                                               double *value,
                                               long *error );
    @brief    Reads a real number value

    @param    fd (IN):
                File descriptor

    @param    element (IN):
                Element name. A NULL value means an iteration.
    @param    value   (OUT):
                Element value
    @param    error   (OUT):
                Error code

    This function reads the value of the next element in the file referred by
    the \a fd whose name matches \a element and returns it as
    a double.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NOT_A_DOUBLE
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_read_real_element_value(long* fd,
                                       char* element,
                                       double* value,
                                       long* error);

  /*! @fn       void xf_tree_read_real_element_array ( long *fd,
                                                     char *element,
                                                     double **array,
                                                     long *length,
                                                     long *error );
    @brief    Reads an array of real number values

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name
    @param    array  (OUT):
                Array of values
    @param    length (OUT):
                Actual array size
    @param    error  (OUT):
                Error code

    This function reads the values of all elements in the file referred by
    the \a fd whose name match \a element and returns them as
    an array of doubles. Please noticed that in case there are several
    conversion errors in the array elements, only the last one is
    published.
    The output array is allocated internally, so the associated memory
    must be freed by the user.

    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_A_DOUBLE
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_read_real_element_array(long* fd,
                                       char* element,
                                       double** array, // AN-853
                                       long* length,
                                       long* error);

  /*! @fn       void xf_tree_read_string_element_value ( long *fd,
                                                       char *element,
                                                       char **value,
                                                       long *error );
    @brief    Reads a string value

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name. If it is NULL, means it is an iteration.
    @param    value   (OUT):
                Element value
    @param    error   (OUT):
                Error code

    This function reads the value of the next element in the file referred by
    the \a fd whose name matches \a element and returns it as a
    characters string.

    - Assumptions :
       - It is assumed that \a *value has no memory reserved and points to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for \a *value. The user must free this memory after using it.
 */
  void xf_tree_read_string_element_value(long* fd,
                                         char* element,
                                         char** value,
                                         long* error);

  /*! @fn void xf_tree_read_string_element_array ( long *fd,
                                          char *element,
                                          char ***array,
                                          long *length,
                                          long *error );
    @brief    Reads an array of string values

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name
    @param    array  (OUT):
                Array of values
    @param    length (OUT):
                Actual array size
    @param    error  (OUT):
                Error code

    This function reads the values of all elements in the file referred by
    the \a fd whose name match \a element and returns them as
    an array of characters strings.
    The output array is allocated internally, so the associated memory
    must be freed by the user.

    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
    - Memory allocation Note: Memory is allocated internally for each \a *array element . The user must free this memory after using it.
 */
  void xf_tree_read_string_element_array(long* fd,
                                         char* element,
                                         char*** array,
                                         long* length,
                                         long* error);

  /*! @fn       void xf_tree_read_string_attribute ( long *fd,
                                                   char *element,
                                                   char *attribute_name,
                                                   char **attribute_value,
                                                   long *error );

    @brief    Reads an attribute as string.

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name. A value of NULL means that the operation is
                performed on the current element name
    @param    attribute_name (IN):
                Attribute name
    @param    attribute_value   (OUT):
                Attribute value
    @param    error   (OUT):
                Error code

    This function reads the value of the attribute with name \a attribute_name,
    from the element specified in \a element (if NULL, it is assumed it has
    already been set) in the file referred by the \a fd, and returns it
    as a characters string. If the node name is not equal to the one set for
    searching, the tree pointer is moved forward to the next element requested.

    - Assumptions :
       - It is assumed that \a *attribute_value has no memory reserved and points to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NO_ATTRIBUTES
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for \a *attribute_value. The user must free this memory after using it.
 */
  void xf_tree_read_string_attribute(long* fd,
                                     char* element,
                                     char* attribute_name,
                                     char** attribute_value,
                                     long* error);

  /*! @fn       void xf_tree_read_integer_attribute ( long *fd,
                                                    char *element,
                                                    char *attribute_name,
                                                    long *attribute_value,
                                                    long *error );

    @brief    Reads an attribute as a long.

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name. If it is NULL, means it is on the current
                element name
    @param    attribute_name (IN):
                Attribute name
    @param    attribute_value   (OUT):
                Attribute value
    @param    error   (OUT):
                Error code

    This function reads the value of the attribute with name \a attribute_name,
    from the element specified in \a element (if NULL, it is assumed it has
    already been set) in the file referred by the \a fd, and returns it
    as a characters long. If the node name is not equal to the one set for
    searching, the tree pointer is moved forward to the next element requested.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NO_ATTRIBUTES
       - XF_CFI_NOT_A_LONG
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_read_integer_attribute(long* fd,
                                      char* element,
                                      char* attribute_name,
                                      long* attribute_value,
                                      long* error);

  /*! @fn       void xf_tree_read_real_attribute ( long *fd,
                                                 char *element,
                                                 char *attribute_name,
                                                 double *attribute_value,
                                                 long *error );

    @brief    Reads an attribute as a double.

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name. If it is NULL, means it is on the current
                element name
    @param    attribute_name (IN):
                Attribute name
    @param    attribute_value   (OUT):
                Attribute value
    @param    error   (OUT):
                Error code

    This function reads the value of the attribute with name \a attribute_name,
    from the element specified in \a element (if NULL, it is assumed it has
    already been set) in the file referred by the \a fd, and returns it
    as a characters double. If the node name is not equal to the one set for
    searching, the tree pointer is moved forward to the next element requested.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NO_ATTRIBUTES
       - XF_CFI_NOT_A_DOUBLE
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_read_real_attribute(long* fd,
                                   char* element,
                                   char* attribute_name,
                                   double* attribute_value,
                                   long* error);

  /*! @fn       void xf_tree_path_read_string_node_value ( long *fd,
                                                         char *path,
                                                         char **value,
                                                         long *error );

    @brief    Reads a string value given the path to the node

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path to the element.
    @param    value (OUT):
                Element value.
    @param    error  (OUT):
                Error code

    This function gets the value of a node (element or attribute) specified
    with an XPath, as string, from the file referred by the \a fd.

    - Assumptions :
       - It is assumed that \a *value has no memory reserved and points to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for \a *value. The user must free this memory after using it.
 */
  void xf_tree_path_read_string_node_value(long* fd,
                                           char* path,
                                           char** value,
                                           long* error);

  /*! @fn       void xf_tree_path_read_integer_node_value ( long *fd,
                                                          char *path,
                                                          long *value,
                                                          long *error );

    @brief    Reads a integer value given the path to the node

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path to the element.
    @param    value (OUT):
                Element value.
    @param    error  (OUT):
                Error code

    This function gets the value of a node (element or attribute) specified
    with an XPath, as long, from the file referred by the \a fd.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_A_LONG
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_path_read_integer_node_value(long* fd,
                                            char* path,
                                            long* value,
                                            long* error);

  /*! @fn       void xf_tree_path_read_real_node_value ( long *fd,
                                                       char *path,
                                                       double *value,
                                                       long *error );

    @brief    Reads a double value given the path to the node

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path to the element.
    @param    value (OUT):
                Element value.
    @param    error  (OUT):
                Error code

    This function gets the value of a node (element or attribute) specified
    with an XPath, as double, from the file referred by the \a fd.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_A_DOUBLE
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_path_read_real_node_value(long* fd,
                                         char* path,
                                         double* value,
                                         long* error);

  /*! @fn void xf_tree_path_read_string_node_array ( long *fd,
                                                   char *path,
                                                   char ***array,
                                                   long *length,
                                                   long *error );

    @brief    Reads an array of string characters given the Xpath to the node

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path to the element. It must reference a list of elements with the
  characters '[*]', which must appear once and only once in the path
    @param    array  (OUT):
                Array of values
    @param    length (OUT):
                Actual array size
    @param    error  (OUT):
                Error code

    This function reads the values of all elements in the file referred by
    the \a fd whose Xpath matches \a path and returns them as
    an array of strings (character arrays). Please noticed that in case
    there are several conversion errors in the array elements, only the
    last one is published.
    The output array is allocated internally, so the associated memory
    must be freed by the user.

    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
    - Memory allocation Note: Memory is allocated internally for each \a *array element . The user must free this memory after using it.
 */
  void xf_tree_path_read_string_node_array(long* fd,
                                           char* path,
                                           char*** array,
                                           long* length,
                                           long* error);

  /*! @fn void xf_tree_path_read_integer_node_array ( long *fd,
                                                    char *path,
                                                    long **array,
                                                    long *length,
                                                    long *error );

    @brief    Reads an array of long values given the Xpath to the node

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path to the element. It must reference a list of elements with the
  characters '[*]', which must appear once and only once in the path
    @param    array  (OUT):
                Array of values
    @param    length (OUT):
                Actual array size
    @param    error  (OUT):
                Error code

    This function reads the values of all elements in the file referred by
    the \a fd whose Xpath matches \a path and returns them as
    an array of longs. Please noticed that in case there are several
    conversion errors in the array elements, only the last one is
    published.
    The output array is allocated internally, so the associated memory
    must be freed by the user.

    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_A_LONG
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_path_read_integer_node_array(long* fd,
                                            char* path,
                                            long** array, // AN-853
                                            long* length,
                                            long* error);

  /*! @fn       void xf_tree_path_read_real_node_array ( long *fd,
                                                       char *path,
                                                       double *array,
                                                       long *length,
                                                       long *error );

    @brief    Reads an array of double values given the Xpath to the node

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path to the element. It must reference a list of elements
                with the characters '[*]', which must appear once and only
                once in the path
    @param    array  (OUT):
                Array of values
    @param    length (OUT):
                Actual array size
    @param    error  (OUT):
                Error code

    This function reads the values of all elements in the file referred by
    the \a fd whose Xpath matches \a path and returns them as
    an array of doubles. Please noticed that in case there are several
    conversion errors in the array elements, only the last one is
    published.
    The output array is allocated internally, so the associated memory
    must be freed by the user.
    
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_A_DOUBLE
       - XF_CFI_VALUE_OUT_OF_RANGE
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */

  void xf_tree_path_read_real_node_array(long* fd,
                                         char* path,
                                         double** array, // AN-853
                                         long* length,
                                         long* error);

  /*! @fn void xf_tree_get_namespace(long  *fd,
                                   char  *node_name, 
                                   long  *num_ns,
                                   char  ***prefix, 
                                   char  ***url,
                                   long  *error); 

    @brief     Reads all the namespace that applies to a given node

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path/node name to the element. It can be the whole path name of the 
                node or just the following node with the given name. If an empty string
                is provided, then the namespace is searched in the root element
    @param    num_ns  (OUT):
                number of namespaces found 
    @param    prefix  (OUT):
                Array of output prefix. 
    @param    url (OUT):
                array with the URL
    @param    error  (OUT):
                Error code

    Reads all the namespace that applies to a given node

    - Assumptions : 
       - There are no more than 50 namespaces in that node
    - Errors   :
       - XF_CFI_VALUE_NOT_FOUND
    - Warnings   :
       - 
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_get_namespace(long* fd,
                             char* node_name,
                             long* num_ns,
                             char*** prefix, // AN-853
                             char*** url, // AN-853
                             long* error);

  /* Navigation functions */
  /* ==================== */

  /*! @fn void xf_tree_rewind ( long *fd, long *error );

    @brief  Sets the read pointer to the beginning of the file

    @param    fd (IN):
                File descriptor
    @param    error(OUT):
                Error code

    Sets the parser pointer to the root element of the document being parsed,
    which is referred by the \a fd.

    - Assumptions : 
       - If the root node is NULL this function does not care
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_rewind(long* fd, long* error);

  /*! @fn void xf_tree_go_to_path_node ( long *fd,
                                       char *path,
                                       long *error );


    @brief  Goes to the node specified in the input XPath

    @param    fd (IN):
                File descriptor
    @param    path (IN):
                Path to the element.
    @param    error(OUT):
                Error code

    Sets the current tree pointer of the file referred by \a fd
    to the node specified in \a path.

    - Assumptions : 
       - If the last node of the path is an attribute, it is not
         taken into account, so that the tree pointer is set to the
         previous element node.
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_go_to_path_node(long* fd,
                               char* path,
                               long* error);

  /*! @fn void xf_tree_go_to_element_node ( long *fd,
                                          char *element,
                                          long *error );


    @brief  Goes to the next element in the tree

    @param    fd (IN):
                File descriptor
    @param    element (IN):
                Element name. If it is NULL, means it is on the current
                element name
    @param    error(OUT):
                Error code

    Sets the current tree pointer of the file referred by \a fd
    to the \a element node.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_go_to_element_node(long* fd,
                                  char* element,
                                  long* error);

  /*! @fn void xf_tree_go_to_next_element_node ( long *fd,
                                               long *error );


    @brief  Goes to the next element in the tree

    @param    fd (IN):
                File descriptor
    @param    error(OUT):
                Error code

    Sets the current tree pointer of the file referred by \a fd
    to the next XML element node in the tree, if not NULL.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_go_to_next_element_node(long* fd,
                                       long* error);

  /*! @fn void xf_tree_get_path ( long *fd,
                                char *path,
                                long *error );


    @brief  Gets the XPath of the current tree pointer

    @param    fd (IN):
                File descriptor
    @param    path (OUT):
                Path of the current element node.
    @param    error(OUT):
                Error code

    Gets the Xpath of the current tree pointer of the file referred by
    \a fd.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_PATH_TOO_LONG
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_get_path(long* fd,
                        char* path,
                        long* error);

  /* Writing functions */
  /* ================= */

  /* XML document and node creation functions */

  /*! @fn     long xf_tree_create( long *error );

    @brief  Create a memory representation of an empty XML document

    @param  error (OUT):
              Error code
    @return A descriptor to be used in all functions for access, insertion
            and removal of element and attributes from the memory representation
            of the XML document, and for writing its content to a file.

    This function must be called first by the user application in order to
    create an XML file from scratch

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_MAX_FILES_REACHED
       - XF_CFI_MEMORY_ERROR
       - XF_CFI_CREATING_DOC_ERROR
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  long xf_tree_create(long* error);

  /*! @fn     void xf_tree_create_root ( long *id,
                                       char *name,
                                       long *error );

    @brief  Create the root element of an XML document

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  name (IN):
              Root element name
    @param  error (OUT):
              Error code

    This function must be called first by the user application in order to
    create an XML file from scratch

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_ROOT_ALREADY_EXISTS
       - XF_CFI_CREATING_NODE_ERROR
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_create_root(long* id,
                           char* name,
                           long* error);

  /*! @fn     void xf_tree_add_child ( long *id,
                                     char *parent,
                                     char *name,
                                     long *error );

    @brief  Add a new element to an XML document as a child of parent

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  parent (IN):
              Parent XPath name
    @param  name (IN):
              New element name
    @param  error (OUT):
              Error code

    This function adds a new element to an XML document at the end of the
    \a parent child list

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_CREATING_NODE_ERROR
       - XF_CFI_ADDING_SIBLING_ERROR
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_add_child(long* id,
                         char* parent,
                         char* name,
                         long* error);

  /*! @fn     void xf_tree_add_next_sibling ( long *id,
                                            char *current,
                                            char *name,
                                            long *error );

    @brief  Add a new element to an XML document after current

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  current (IN):
              XPath name of the element used as reference for the
              new element insertion
    @param  name (IN):
              New element name
    @param  error (OUT):
              Error code

    This function adds a new element to an XML document after the \a current
    element

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_CREATING_NODE_ERROR
       - XF_CFI_ADDING_SIBLING_ERROR
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_add_next_sibling(long* id,
                                char* current,
                                char* name,
                                long* error);

  /*! @fn     void xf_tree_add_previous_sibling ( long *id,
                                                char *current,
                                                char *name,
                                                long *error );

    @brief  Add a new element to an XML document before current

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  current (IN):
              XPath name of the element used as reference for the
              new element insertion
    @param  name (IN):
              New element name
    @param  error (OUT):
              Error code

    This function adds a new element to an XML document before the \a current
    element

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_CREATING_NODE_ERROR
       - XF_CFI_ADDING_SIBLING_ERROR
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_add_previous_sibling(long* id,
                                    char* current,
                                    char* name,
                                    long* error);

  /*! @fn     void xf_tree_add_attribute ( long *id,
                                         char *current,
                                         char *name,
                                         long *error );

    @brief  Add a new attribute carried by an element

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  current (IN):
              XPath name of the element
    @param  name (IN):
              New attribute name
    @param  error (OUT):
              Error code

    This function adds a new attribute to the \a current element

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_add_attribute(long* id,
                             char* current,
                             char* name,
                             long* error);

  /*! @fn     void xf_tree_remove_node ( long *id,
                                       char *name,
                                       long *error );

    @brief  Remove a node

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  name (IN):
              XPath node name. It may be either an element or an attribute
    @param  error (OUT):
              Error code

    This function unlinks the \a name node and all its descendants from the
    memory representation of the XML document.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_REMOVING_NODE_ERROR
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_remove_node(long* id,
                           char* name,
                           long* error);

  /*! @fn     void xf_tree_set_integer_node_value ( long *id,
                                                  char *name,
                                                  long *value,
                                                  char *format,
                                                  long *error );

    @brief  Set an integer node value

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  name (IN):
              XPath node name. It may be either an element or an attribute
    @param  value (IN):
              Integer value
    @param  format (IN):
              Integer representation format
    @param  error (OUT):
              Error code

    Replace the value of the \a name node by the integer \a value.
    \a format may be any valid format specification accepted by the printf
    family of C functions, i.e. "%+010d"

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_INVALID_FORMAT
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_set_integer_node_value(long* id,
                                      char* name,
                                      long* value,
                                      char* format,
                                      long* error);

  /*! @fn     void xf_tree_set_real_node_value ( long *id,
                                               char *name,
                                               double *value,
                                               char *format,
                                               long *error );

    @brief  Set a real number node value

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  name (IN):
              XPath node name. It may be either an element or an attribute
    @param  value (IN):
              Real number value
    @param  format (IN):
              Real number representation format
    @param  error (OUT):
              Error code

    Replace the value of the \a name node by the double \a value.
    \a format may be any valid format specification accepted by the printf
    family of C functions, i.e. "%+012.6lf"

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_INVALID_FORMAT
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_set_real_node_value(long* id,
                                   char* name,
                                   double* value,
                                   char* format,
                                   long* error);

  /*! @fn     void xf_tree_set_string_node_value ( long *id,
                                                 char *name,
                                                 char *value,
                                                 char *format,
                                                 long *error );

    @brief  Set a string node value

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  name (IN):
              XPath node name. It may be either an element or an attribute
    @param  value (IN):
              String value
    @param  format (IN):
              String representation format
    @param  error (OUT):
              Error code

    Replace the value of the \a name node by the string \a value.
    \a format may be any valid format specification accepted by the printf
    family of C functions, i.e. "%10s"

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_INVALID_FORMAT
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_set_string_node_value(long* id,
                                     char* name,
                                     char* value,
                                     char* format,
                                     long* error);

  /*! @fn     void xf_tree_write ( long *id,
                                 char *name,
                                 long *error );

    @brief  Write the data to a file on disk.

    @param  id (IN):
              Descriptor of the XML document memory representation
    @param  name (IN):
              File name. If it is "-", then the standard output is used
    @param  error (OUT):
              Error code

    Write an XML document previously opened or created to a file on disk.

    The empty tags in a XML file can be written in two ways:
      - short format: <tag/>
      - long format:  <tag></tag>
    The used format depends on the libxml's global variable "xmlSaveNoEmptyTags"
    (defined in libxml/globals.h). This variable is set to 0 by default (so that 
    the short format is written). To use the long format, the variable has to be 
    set to 1.

    - Assumptions : N/A
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_CREATING_NODE_ERROR
       - XF_CFI_ADDING_SIBLING_ERROR
       - XF_CFI_SAVING_DOC_ERROR
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_write(long* id,
                     char* name,
                     long* error);

  /*! @fn void xf_set_schema (char *filename,
                            char *schema, 
                            long *action,
                            long *error);
 
    @brief It sets the schema in the root element of the filename.
 
    @par   filename (IN):
             filename
    @par   schema (IN):
             schema path.
    @par   action (IN):
             flag to indicate what to do:
               - XF_CHANGE_SCHEMA_AND_VERSION: Change schema and version
               - XF_CHANGE_SCHEMA:             Change schema
               - XF_DELETE_ALL:                Delete schema and version attributes
               - XF_DELETE_SCHEMA:             Delete schema attribute
    @par    error(OUT):
              Error code.
  
    This function sets the schema in the root element of the filename
 
    - Assumptions : N/A
    - Errors   : 
    - XF_CFI_PARSING_FILE_ERROR
    - XF_CFI_SAVING_DOC_ERROR
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */

  void xf_set_schema(char* filename, char* schema, long* action, long* error);

  /* Search functions */
  /* ================ */

  /*! @fn     void xf_tree_find_string_value_element( long *fd,
                                                    char *element,
                                                    char *value,
                                                    char *found_path,
                                                    long *error );

    @brief  Find the XPath name of an element given an element name
            and value
    
    @param  fd (IN):
              File descriptor
    @param  element (IN):
              Element name used for the search operation
    @param  value (IN):
              Value
    @param  found_path (OUT):
              XPath expression of the found element
    @param  error (OUT):
              Error code

    This function finds the XPath name of the next element whose name and
    value match the \a element and \a value provided as input
    
    This function iterates through all nodes in the file, starting from the
    current location
    
    The search operation is based on the exact lexicographical match of both
    the element name and the value 
    
    - Assumptions :
       - It is assumed that \a *found_path has no memory reserved and points to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_VALUE_NOT_FOUND
    - Warnings   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for \a *found_path. The user must free this memory after using it.
 */
  void xf_tree_find_string_value_element(long* fd,
                                         char* element,
                                         char* value,
                                         char* found_path,
                                         long* error);

  /*! @fn     void xf_tree_find_string_value_path( long *fd,
                                                 char *path,
                                                 char *value,
                                                 char *found_path,
                                                 long *error );

    @brief  Find the XPath name of an element given an XPath expression
            and value
    
    @param  fd (IN):
              File descriptor
    @param  path (IN):
              XPath expression used for the search operation
    @param  value (IN):
              Value
    @param  found_path (OUT):
              XPath expression of the found element
    @param  error (OUT):
              Error code

    This function finds the XPath name of the first element whose XPath
    name matches the \a path expression and whose value matches 
    the \a value provided as input.
    
    The XPath expression in the \a path input parameter must reference a
    list of elements with the characters '[*]', which must appear once
    and only once in the path.
    
    This function is intended for searching the XPath expression of an
    element within a list whose value is unique. For a more general search,
    use the \a xf_tree_find_string_value_element function instead.
    
    The search operation is based on the exact lexicographical match of both
    the element path name and the value.
    
    - Assumptions :
       - It is assumed that \a *found_path has no memory reserved and points to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_ROOT_NODE_IS_NULL
       - XF_CFI_CURRENT_NODE_IS_NULL
       - XF_CFI_NOT_AN_ELEMENT_NODE
       - XF_CFI_BAD_INPUT_ARGUMENT
       - XF_CFI_NO_ELEMENT_REQUESTED
       - XF_CFI_PATH_TOO_LONG
       - XF_CFI_NODE_NAME_TOO_LONG
       - XF_CFI_NOT_A_TERMINAL_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
    - Warnings   :
       - XF_CFI_ROOT_ALREADY_REACHED
       - XF_CFI_MAX_LENGTH_EXCEEDED
    - Reference : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for \a *found_path. The user must free this memory after using it.
 */
  void xf_tree_find_string_value_path(long* fd,
                                      char* path,
                                      char* value,
                                      char* found_path,
                                      long* error);

  /* Error functions */
  /* =============== */

  /*! @fn  void xf_basic_error_msg ( long error_code, char *error_message );

    @brief  Gets default message corresponding to the input error code

    @param    error_code   (IN):
                Code of the error or warning to be read.
    @param    error_message   (OUT):
                Error message associated to the input code.

    This function returns the error message associated with the input error
    code. If the error code is unknown, an adequate error message is
    returned.

    - Assumptions :
       - It is assumed that the size of the \a error_message
         variable is equal or greater than
         \a XF_MAX_ERROR_MSG_LENGTH
    - Errors   : N/A
    - Warnings   : N/A
    - Reference   : N/A
 */
  void xf_basic_error_msg(long error_code, char* error_message);

  /* Error codes: The first element must be updated when adding new error codes    */
  /* ============================================================================= */

  /*! @enum  XF_CFI_General_err_enum

    @brief List of error and warning codes

    @li OK      : code = 0
    @li Warning : code > 0
    @li Error   : code < 0

 */

  void xf_verbose(); //AN-584
  /*! @fn  void xf_verbose ( );

    @brief  Set verbosity flag

    This function set the verbosity flag. When the flag is set the errors
    raised by explorer_file_handling functions as well as the ones raised by
    libxml are displayed to stdedd.
 */

  void xf_silent(); //AN-584
  /*! @fn  void xf_silent ( );

    @brief  Unset verbosity flag

    This function unset the verbosity flag. When the flag is unset the errors
    raised by explorer_file_handling functions as well as the ones raised by
    libxml are NOT displayed to stdedd.
 */

  typedef enum
  {
    XF_CFI_FIRST_ELEMENT = -39, /*!< First element. AN-853: removed one error code*/

    XF_CFI_LOCK_ERROR, /*!< Could not lock other running threads */
    XF_CFI_UNLOCK_ERROR, /*!< Could not unlock other blocked threads */

    XF_CFI_VALUE_NOT_FOUND, /*!< The requested value has not been found */
    XF_CFI_FILENAME_TOO_LONG, /*!< The filename is too long*/
    XF_CFI_WRONG_HEADER_ELEMENT, /*!< Element not allowed to be included in a header*/
    XF_CFI_WRONG_HEADER_FILE_EXTENSION, /*!< File can?t contain a header*/
    XF_CFI_WRONG_HEADER_FILE_FORMAT, /*!< The header file contains unwanted tags*/

    XF_CFI_EMPTY_ELEMENT, /*!< Element is empty*/
    XF_CFI_NO_ARRAY_PATH, /*!< The given path is not for an array*/
    XF_CFI_SAVING_DOC_ERROR, /*!< Unable to save the XML document into disk*/
    XF_CFI_INVALID_FILE_FORMAT, /*!< Unable to read an item. Invalid file format*/

    XF_CFI_INVALID_FORMAT, /*!< Printing format provided is not valid*/
    XF_CFI_REMOVING_NODE_ERROR, /*!< Unable to remove an element node*/

    XF_CFI_ADDING_SIBLING_ERROR, /*!< Unable to add a sibling node*/
    XF_CFI_CREATING_NODE_ERROR, /*!< Unable to create an element node*/
    XF_CFI_ROOT_ALREADY_EXISTS, /*!< Root element already exists*/
    XF_CFI_CREATING_DOC_ERROR, /*!< Error when creating a new XML document*/
    XF_CFI_WRONG_FILE_DESCRIPTOR, /*!< File descriptor out of range*/

    XF_CFI_CURRENT_NODE_IS_NULL, /*!< Current node is NULL. Rewind is needed*/
    XF_CFI_MAX_FILES_REACHED, /*!< The max. number of open files has been
                                              reached. No more files are allowed*/
    XF_CFI_NODE_NAME_TOO_LONG, /*!< Name of node exceeds the maximum allowed*/
    XF_CFI_PATH_TOO_LONG, /*!< Path length exceeds the maximum allowed*/
    XF_CFI_NOT_A_TERMINAL_ELEMENT, /*!< Current node has other xml
                                                elements as value*/

    XF_CFI_NO_ATTRIBUTES, /*!< Current node has no attributes*/
    XF_CFI_VALUE_OUT_OF_RANGE, /*!< Element value (integer or real)
                                                is out of range*/
    XF_CFI_NOT_A_DOUBLE, /*!< Value to be converted is not a double*/
    XF_CFI_NOT_A_LONG, /*!< Value to be converted is not an integer*/
    XF_CFI_NO_ELEMENT_FOUND, /*!< No element was found in the search*/

    XF_CFI_ROOT_NODE_IS_NULL, /*!< Root node is NULL. Initialization is needed*/
    XF_CFI_NOT_AN_ELEMENT_NODE, /*!< Current node is not an xml element node*/
    XF_CFI_NO_ELEMENT_REQUESTED, /*!< There is no element name set to be searched*/
    XF_CFI_DOC_NOT_PARSED_OR_CREATED, /*!< No XML document has been parsed or created*/
    XF_CFI_MEMORY_ERROR, /*!< Unable to ask for memory*/

    XF_CFI_GETTING_ROOT_ERROR, /*!< Unable to get the root element
                                                during initialisation*/
    XF_CFI_PARSING_FILE_ERROR, /*!< Error during initialisation*/
    XF_CFI_BAD_INPUT_ARGUMENT, /*!< Bad input argument*/
    XF_CFI_GENERIC_ERROR, /*!< Generic error*/
    XF_CFI_UNKNOWN_ERROR_CODE, /*!< Error code for unknown error codes*/

    XF_CFI_OK = 0, /*!< OK */

    XF_CFI_ROOT_ALREADY_REACHED, /*!< Warning used when the path goes
                                                up too much*/
    XF_CFI_MAX_LENGTH_EXCEEDED, /*!< Length of element name or value
                                                exceeds the size of the storage variable*/
    XF_CFI_GENERIC_WARNING, /*!< Generic warning*/
    XF_CFI_HEADER_ALREADY_EXISTS, /*!< Header already exists*/

    XF_CFI_LAST_ELEMENT /*!< Last element*/
  } XF_CFI_General_err_enum;

  /* ===================================================== */
  /* NEW FUNCTIONS FOR WRITING EARTH EXPLORER FILE HEADERS */
  /* ===================================================== */
  /*#ifdef XF_EARTH_EXPLORER_HEADER*/
  /*!
    @fn void xf_tree_create_header (long *fd,
                                    long *file_extension_type,
                                    long *error)

    @brief Generates a new blank Header including both Fixed and
           Variable headers

    @param fd (IN):
             File descriptor
    @param file_extension_type (IN):
             Type of XML file to add Header to.
             See \a XF_File_extension_type_enum
    @param error (OUT):
             Error code

    This function writes the complete Cryosat Header (full Fixed Header +
    Variable Header).
    If the XML file already has a Header this function shall not rewrite
    it again.

    - Assumptions :
       - XF_EARTH_EXPLORER_HEADER macro is defined
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_CREATING_NODE_ERROR
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_WRONG_HEADER_FILE_EXTENSION
       - XF_CFI_HEADER_ALREADY_EXISTS
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_create_header(long* fd, long* file_extension_type, long* error);

  /*!
    @fn void xf_tree_set_fixed_header_item ( long *fd,
                                             char *item_id,
                                             char *item_value,
                                             long *error);

    @brief Sets the value of any element of a Header

    @param fd (IN):
             File descriptor
    @param item_id (IN):
             Name of the element which value we want to modify
    @param item_value (IN):
             New value assigned for the node
    @param error (OUT):
             Error code

    This function sets the value of a Fixed Header node.

    - Assumptions :
       - XF_EARTH_EXPLORER_HEADER macro is defined
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_WRONG_HEADER_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_INVALID_FORMAT
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_set_fixed_header_item(long* fd,
                                     char* item_id,
                                     char* item_value,
                                     long* error);

  /*!
    @fn void xf_tree_set_fixed_header_items ( long *fd,
                                     char *file_name,
                                     char *file_description,
                                     char *notes,
                                     char *mission,
                                     char *file_class,
                                     char *file_type,
                                     char *validity_start,
                                     char *validity_stop,
                                     long *file_version,
                                     char *system,
                                     char *creator,
                                     char *creator_version,
                                     char *creation_date,
                                     long *error);

    @brief Sets the value of all Fixed Header elements

    @param fd    (IN):
             File descriptor
    @param file_name   (IN):
             Value of the element file_name
    @param file_description  (IN):
             Value of the element file_description
    @param notes   (IN):
             Value of the element notes
    @param mission (  IN):
             Value of the element mission
    @param file_class   (IN):
             Value of the element file_class
    @param file_type   (IN):
             Value of the element file_type
    @param validity_start  (IN):
             Value of the element validity_start
    @param validity_stop  (IN):
             Value of the element validity_stop
    @param file_version  (IN):
             Value of the element file_version
    @param system   (IN):
             Value of the element system
    @param creator  (IN):
             Value of the element creator
    @param creator_version  (IN):
             Value of the element creator_version
    @param creation_date  (IN):
             Value of the element creation_date
    @param error   (OUT):
             Error code

    This function sets the value of a Fixed Header node.

    - Assumptions :
       - XF_EARTH_EXPLORER_HEADER macro is defined
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_INVALID_FORMAT
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
 */
  void xf_tree_set_fixed_header_items(long* fd, char* file_name, char* file_description, char* notes, char* mission, char* file_class, char* file_type, char* validity_start, char* validity_stop, long* file_version, char* system, char* creator, char* creator_version, char* creation_date, long* error);

  /*!
    @fn void xf_tree_get_fixed_header_item ( long *fd,
                                             char *item_id,
                                             char **item_value,
                                             long *error);

    @brief Gets the value of any element of a Header

    @param fd (IN):
             File descriptor
    @param item_id (IN):
             Name of the element which value we want to retrieve
    @param item_value (OUT):
             Value of the requested element
    @param error (OUT):
             Error code

    This function gets the value of a Fixed Header node.

    - Assumptions :
       - XF_EARTH_EXPLORER_HEADER macro is defined
       - It is assumed that \a *item_value has no memory reserved and points to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_WRONG_HEADER_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_INVALID_FILE_FORMAT
       - XF_CFI_EMPTY_ELEMENT
    - Warnings   : N/A
    - Reference   : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for \a *item_value. The user must free this memory after using it.
 */
  void xf_tree_get_fixed_header_item(long* fd,
                                     char* item_id,
                                     char** item_value,
                                     long* error);

  /*!
    @fn void xf_tree_get_fixed_header_items ( long *fd,
                                              char **file_name,
                                              char **file_description,
                                              char **notes,
                                              char **mission,
                                              char **file_class,
                                              char **file_type,
                                              char **validity_start,
                                              char **validity_stop,
                                              long *file_version,
                                              char **system,
                                              char **creator,
                                              char **creator_version,
                                              char **creation_date,
                                              long *error);

    @brief Gets the value of all Fixed Header elements

    @param fd    (IN):
             File descriptor
    @param file_name   (OUT):
             Value of the element file_name
    @param file_description  (OUT):
             Value of the element file_description
    @param notes   (OUT):
             Value of the element notes
    @param mission (  OUT):
             Value of the element mission
    @param file_class   (OUT):
             Value of the element file_class
    @param file_type   (OUT):
             Value of the element file_type
    @param validity_start  (OUT):
             Value of the element validity_start
    @param validity_stop  (OUT):
             Value of the element validity_stop
    @param file_version  (OUT):
             Value of the element file_version
    @param system   (OUT):
             Value of the element system
    @param creator  (OUT):
             Value of the element creator
    @param creator_version  (OUT):
             Value of the element creator_version
    @param creation_date  (OUT):
             Value of the element creation_date
    @param error   (OUT):
             Error code

    This function sets the value of a Fixed Header node.

    - Assumptions :
       - XF_EARTH_EXPLORER_HEADER macro is defined
       - All (char*) arguments have no memory reserved and point to NULL
    - Errors   :
       - XF_CFI_WRONG_FILE_DESCRIPTOR
       - XF_CFI_DOC_NOT_PARSED_OR_CREATED
       - XF_CFI_WRONG_HEADER_ELEMENT
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_INVALID_FORMAT
    - Warnings   : N/A
    - Reference  : See the \a libxml documentation
    - Memory allocation Note: Memory is dynamically allocated internally for each char* argument. The user must free this memory after using it.

 */
  void xf_tree_get_fixed_header_items(long* fd,
                                      char** file_name,
                                      char** file_description,
                                      char** notes,
                                      char** mission,
                                      char** file_class,
                                      char** file_type,
                                      char** validity_start,
                                      char** validity_stop,
                                      long* file_version,
                                      char** system,
                                      char** creator,
                                      char** creator_version,
                                      char** creation_date,
                                      long* error);

  /*!
    @fn void xf_create_filename ( long *mission_id,
                                  char *file_class,
                                  char *file_type,
                                  char *instance_id,
                                  long *file_extension_type,
                                  char *filename,
                                  long *error);

    @brief Generates a complete Cryosat File Name

    @param mission_id (IN):
             Satellite ID to add the following to the filename.
             See \a XF_Sat_id_enum
    @param file_class (IN):
             Type of activity for which the file is used
    @param file_type (IN):
             File Type
    @param instance_id (IN):
             Makes the file unique
    @param file_extension_type (IN):
             Type of XML file to add Header.
             See \a XF_File_extension_type_enum
    @param filename (OUT):
             Filename generated
    @param error (OUT):
             Error code

    This function generates a complete Cryosat File Name

    - Assumptions :
       - XF_EARTH_EXPLORER_HEADER macro is defined
    - Errors   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
       - XF_CFI_BAD_INPUT_ARGUMENT
    - Warnings   : N/A
    - Reference   : None
 */
  void xf_create_filename(long* satellite_id,
                          char* file_class,
                          char* file_type,
                          char* instance_id,
                          long* file_extension_type,
                          char* filename,
                          long* error);

  /*!
    @fn  void xf_read_filename_items ( char *filename,
                                       long *satellite_id,
                                       char *file_class,
                                       char *file_type,
                                       char *instance_id,
                                       long *error);

    @brief Reads a filename and returns its parameters

    @param filename (IN):
             Input filename
    @param satellite_id (OUT):
             Satellite ID corresponding to the file.
             See \a XF_Sat_id_enum
    @param file_class (OUT):
             Type of activity for which the file is used
    @param file_type (OUT):
             File Type
    @param instance_id (OUT):
             Makes the file unique                                                   @param error (OUT):
             Error code

    This function generates a complete Cryosat File Name

    - Assumptions :
       - XF_EARTH_EXPLORER_HEADER macro is defined
    - Errors   :
       - XF_CFI_MAX_LENGTH_EXCEEDED
       - XF_CFI_BAD_INPUT_ARGUMENT
    - Warnings   : N/A
    - Reference   : None
 */
  void xf_read_filename_items(char* filename,
                              long* satellite_id,
                              char* file_class,
                              char* file_type,
                              char* instance_id,
                              long* error);

  /*!
    @fn  xf_copy_node(long *fd, 
		  char *dest_node,
		  char *src_file,
		  char *source_node,
		  long *error);

    @brief Copy a node element from a file in other file

    @param fd (IN):
             Input file descriptor
    @param dest_node (IN):
             Destination node name
    @param src_file (IN):
             Source filename
    @param source_node (IN):
             Node to copy
    @param errr (OUT):
             Error code

    This function copies a node from the src_file into the
    dest_node within the file whose descriptor is fd

    - Assumptions :
    - Errors   :
       - XF_CFI_NO_ELEMENT_FOUND
       - XF_CFI_PARSING_FILE_ERROR
    - Warnings   : N/A
    - Reference   : None
 */
  void xf_copy_node(long* fd,
                    char* dest_node,
                    char* src_file,
                    char* source_node,
                    long* error);

  /*!
    @fn  xf_check_library_version();

    @brief Get the library version

    This function prints in the standard output the library version

    - Assumptions : None
    - Errors      : N/A
    - Warnings    : N/A
    - Reference   : None
 */
  long xf_check_library_version();

#if defined(expcfi_check_libs)

  #if !defined(xd_check_library_version) && !defined(xl_check_library_version) && !defined(xo_check_library_version) && !defined(xp_check_library_version) && !defined(xv_check_library_version)

    #undef expcfi_check_libs
    #define expcfi_check_libs xf_check_library_version

  #endif

#else

  #define expcfi_check_libs xf_check_library_version

#endif

  /*! @enum XF_File_extension_type_enum

    @brief List of XML Header files allowed

 */
  typedef enum
  {
    XF_HEADER_FORMAT_EEF, /*!< XML file with Header and Datablock (Earth Explorer mission) */
    XF_HEADER_FORMAT_HDR, /*!< XML file with Header*/
    XF_HEADER_FORMAT_DBL, /*!< XML file with Datablock*/
    XF_HEADER_FORMAT_NONE, /*!< No extension when generating filename*/
    XF_HEADER_FORMAT_EOF /*!< XML file with Header and Datablock (Earth Observation mission) */
  } XF_File_extension_type_enum;

  /*! @enum XF_Set_schema_enum

    @brief List of actions to do when calling to xf_set_schema

 */
  typedef enum
  {
    XF_CHANGE_SCHEMA_AND_VERSION, /*!< Change schema and version */
    XF_CHANGE_SCHEMA, /*!< Change schema */
    XF_DELETE_ALL, /*!< Delete schema and version attributes */
    XF_DELETE_SCHEMA /*!< Delete schema attribute */
  } XF_Set_schema_enum;

  /*! @enum XF_Sat_id_enum

    @brief List of Satellite Ids allowed

 */

  typedef enum
  {

    XF_SAT_DEFAULT = 0, /*!< Default satellite Id */
    XF_SAT_DEFAULT1 = 1,
    XF_SAT_DEFAULT2 = 2,
    XF_SAT_DEFAULT3 = 3,
    XF_SAT_DEFAULT4 = 4,
    XF_SAT_DEFAULT5 = 5,
    XF_SAT_DEFAULT6 = 6,
    XF_SAT_DEFAULT7 = 7,
    XF_SAT_DEFAULT8 = 8,
    XF_SAT_DEFAULT9 = 9,
    XF_SAT_ERS1 = 11, /*!< Satellite Id for ERS1*/
    XF_SAT_ERS2 = 12, /*!< Satellite Id for ERS2*/
    XF_SAT_ENVISAT = 21, /*!< Satellite Id for Envisat*/
    XF_SAT_METOP1 = 31, /*!< Satellite Id for MetOp1*/
    XF_SAT_METOP2 = 32, /*!< Satellite Id for MetOP2*/
    XF_SAT_METOP3 = 33, /*!< Satellite Id for MetOp3*/
    XF_SAT_CRYOSAT = 41, /*!< Satellite Id for Cryosat*/
    XF_SAT_ADM = 51, /*!< Satellite Id for Aeolus*/
    XF_SAT_GOCE = 61, /*!< Satellite Id for GOCE*/
    XF_SAT_SMOS = 71, /*!< Satellite Id for SMOS*/
    XF_SAT_TERRASAR = 81, /*!< Satellite Id for TERRASAR*/
    XF_SAT_EARTHCARE = 91, /*!< Satellite Id for EARTHCARE */
    XF_SAT_SWARM_A = 101, /*!< Satellite Id for SWARM A */
    XF_SAT_SWARM_B = 102, /*!< Satellite Id for SWARM B */
    XF_SAT_SWARM_C = 103, /*!< Satellite Id for SWARM C */
    XF_SAT_SENTINEL_1A = 110,
    XF_SAT_SENTINEL_1B = 111,
    XF_SAT_SENTINEL_2 = 112, /* deprecated */
    XF_SAT_SENTINEL_3 = 113, /* deprecated */
    XF_SAT_SEOSAT = 120,
    XF_SAT_SENTINEL_1C = 125,
    XF_SAT_SENTINEL_2A = 126,
    XF_SAT_SENTINEL_2B = 127,
    XF_SAT_SENTINEL_2C = 128,
    XF_SAT_SENTINEL_3A = 129,
    XF_SAT_SENTINEL_3B = 130,
    XF_SAT_SENTINEL_3C = 131,
    XF_SAT_JASON_CSA = 132, // AN-531
    XF_SAT_JASON_CSB = 133, // AN-531
    XF_SAT_METOP_SG_A1 = 134, // AN-530
    XF_SAT_METOP_SG_A2 = 135, // AN-530
    XF_SAT_METOP_SG_A3 = 136, // AN-530
    XF_SAT_METOP_SG_B1 = 137, // AN-530
    XF_SAT_METOP_SG_B2 = 138, // AN-530
    XF_SAT_METOP_SG_B3 = 139, // AN-530
    XF_SAT_SENTINEL_5P = 140, // AN-502
    XF_SAT_BIOMASS = 141, // AN-601
    XF_SAT_SENTINEL_5 = 142, // AN-639
    XF_SAT_SAOCOM_CS = 143, // AN-640
    XF_SAT_FLEX = 144, /* AN-683 */
    XF_SAT_SENTINEL_6A = 145, /* AN-809 */
    XF_SAT_SENTINEL_6B = 146, /* AN-809 */
    XF_SAT_CIMR = 147,
    XF_SAT_ROSEL = 148,
    XF_SAT_CHIME = 149,
    XF_SAT_CRISTAL = 150,
    XF_SAT_CO2M = 151,
    XF_SAT_LSTM = 152,
    XF_SAT_FORUM = 153,
    XF_SAT_GENERIC = 200,
    XF_SAT_GENERIC_GEO = 300, /* ANR-353 */
    XF_SAT_MTG = 301, /* ANR-353 */
    XF_SAT_GENERIC_MEO = 400 /* ANR-550 */

  } XF_Sat_id_enum;

  /*#endif*/

  /* ====================================================== */
  /* /NEW FUNCTIONS FOR WRITING EARTH EXPLORER FILE HEADERS */
  /* ====================================================== */

#ifdef __cplusplus
}
#endif

#endif
